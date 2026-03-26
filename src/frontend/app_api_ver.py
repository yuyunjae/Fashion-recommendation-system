from __future__ import annotations
import io
import os
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

# requests import 시 출력되는 dependency warning 숨김
warnings.filterwarnings(
    "ignore",
    message="Unable to find acceptable character detection dependency.*",
)

import requests
from PIL import Image

STYLE_OPTIONS_MULTI_SCRATCH: List[str] = [
    "athleisure",
    "bodyconscious",
    "bold",
    "cityglam",
    "classic",
    "disco",
    "ecology",
    "feminine",
    "genderless",
    "grunge",
    "hiphop",
    "hippie",
    "ivy",
    "kitsch",
    "lingerie",
    "lounge",
    "metrosexual",
    "military",
    "minimal",
    "mods",
    "normcore",
    "oriental",
    "popart",
    "powersuit",
    "punk",
    "space",
    "sportivecasual",
]

COMBINED_LABELS_MULTI_SCRATCH: List[str] = [
    "athleisure_female",
    "bodyconscious_female",
    "bold_male",
    "cityglam_female",
    "classic_female",
    "disco_female",
    "ecology_female",
    "feminine_female",
    "genderless_female",
    "grunge_female",
    "hiphop_female",
    "hiphop_male",
    "hippie_female",
    "hippie_male",
    "ivy_male",
    "kitsch_female",
    "lingerie_female",
    "lounge_female",
    "metrosexual_male",
    "military_female",
    "minimal_female",
    "mods_male",
    "normcore_female",
    "normcore_male",
    "oriental_female",
    "popart_female",
    "powersuit_female",
    "punk_female",
    "space_female",
    "sportivecasual_female",
    "sportivecasual_male",
]
MALE_STYLE_SET = {
    label.rsplit("_", 1)[0]
    for label in COMBINED_LABELS_MULTI_SCRATCH
    if label.endswith("_male")
}
FEMALE_STYLE_SET = {
    label.rsplit("_", 1)[0]
    for label in COMBINED_LABELS_MULTI_SCRATCH
    if label.endswith("_female")
}


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (
            len(value) >= 2
            and ((value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")))
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)


load_env_file(Path(__file__).resolve().parent / ".env")


def normalize_gender(value: str) -> str:
    v = str(value).strip().lower()
    if v in {"man", "men", "m", "male"}:
        return "male"
    if v in {"woman", "women", "w", "f", "female"}:
        return "female"
    return "all"


def join_url(base: str, path: str) -> str:
    base = str(base).rstrip("/")
    path = str(path).strip()
    if not path:
        path = "/search"
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{base}{path}"


def extract_results(payload: object) -> List[Dict[str, object]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("results", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def get_image_ref(item: Dict[str, object]) -> str:
    for key in ("image_url", "url", "thumbnail_url", "image_path", "path"):
        val = item.get(key)
        if val is None:
            continue
        txt = str(val).strip()
        if txt:
            return txt
    return ""


class StreamlitApiRecommendationApp:
    def __init__(self) -> None:
        try:
            import streamlit as st
        except ImportError as exc:
            raise RuntimeError("Streamlit is not installed. Install with: pip install streamlit") from exc
        self.st = st

    def run(self) -> None:
        st = self.st
        st.set_page_config(page_title="Fashion Recommender (API)", layout="wide")
        st.title("Fashion Recommendation (API Client)")
        st.caption("Calls backend /search endpoint and renders results")

        settings = self._load_runtime_settings()
        self._render_main_ui(settings, list(STYLE_OPTIONS_MULTI_SCRATCH))

    def _load_runtime_settings(self) -> Dict[str, object]:
        default_api_base = self._resolve_default("API_BASE_URL", "localhost:8000")
        default_search_path = self._resolve_default("SEARCH_PATH", "/search")
        default_image_path_template = self._resolve_default("IMAGE_PATH_TEMPLATE", "/search_image/{item_id}")
        default_image_id_query_key = self._resolve_default("IMAGE_ID_QUERY_KEY", "item_id")
        default_timeout = self._resolve_default_int("REQUEST_TIMEOUT_SEC", 60)
        default_verify_ssl = self._resolve_default_bool("VERIFY_SSL", False)
        default_preview_mode = self._resolve_default_bool("PREVIEW_MODE", False)

        return {
            "api_base": str(default_api_base).strip(),
            "search_path": str(default_search_path).strip(),
            "image_path_template": str(default_image_path_template).strip(),
            "image_id_query_key": str(default_image_id_query_key).strip(),
            "timeout_sec": max(5, min(600, int(default_timeout))),
            "verify_ssl": bool(default_verify_ssl),
            "preview_mode": bool(default_preview_mode),
        }

    def _style_options_for_gender(
        self,
        gender_filter: str,
        all_style_options: Sequence[str],
    ) -> List[str]:
        normalized = normalize_gender(gender_filter)
        if normalized == "male":
            return [s for s in all_style_options if s in MALE_STYLE_SET]
        if normalized == "female":
            return [s for s in all_style_options if s in FEMALE_STYLE_SET]
        return list(all_style_options)

    def _resolve_default(self, key: str, fallback: str) -> str:
        env_val = os.getenv(key, "").strip()
        return env_val if env_val else fallback

    def _resolve_default_int(self, key: str, fallback: int) -> int:
        raw = self._resolve_default(key, str(fallback))
        try:
            return int(raw)
        except Exception:
            return int(fallback)

    def _resolve_default_bool(self, key: str, fallback: bool) -> bool:
        raw = self._resolve_default(key, "true" if fallback else "false").strip().lower()
        if raw in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if raw in {"0", "false", "f", "no", "n", "off"}:
            return False
        return bool(fallback)

    def _resolve_default_float(self, key: str, fallback: float) -> float:
        raw = self._resolve_default(key, str(fallback))
        try:
            return float(raw)
        except Exception:
            return float(fallback)

    def _build_request(
        self,
        uploaded,
        gender_filter: str,
        top_k: int,
        preferred_styles: Sequence[str],
        disliked_styles: Sequence[str],
        fallback_fill: bool,
        beta_like: float,
        gamma_dislike: float,
        delta_survey_prior: float,
        settings: Dict[str, object],
    ) -> Tuple[str, Dict[str, Tuple[str, bytes, str]], Dict[str, str], Dict[str, str]]:
        url = join_url(str(settings["api_base"]), str(settings["search_path"]))

        content_type = uploaded.type or "application/octet-stream"
        files = {"image": (uploaded.name, uploaded.getvalue(), content_type)}

        data = {
            "gender": normalize_gender(gender_filter),
            "top_k": str(int(top_k)),
            "preferred_styles": ",".join([s.strip().lower() for s in preferred_styles if str(s).strip()]),
            "disliked_styles": ",".join([s.strip().lower() for s in disliked_styles if str(s).strip()]),
            "fallback_fill": "true" if fallback_fill else "false",
            "beta_like": str(float(beta_like)),
            "gamma_dislike": str(float(gamma_dislike)),
            "delta_survey_prior": str(float(delta_survey_prior)),
        }

        headers: Dict[str, str] = {}
        
        return url, files, data, headers

    def _extract_item_id(self, item: Dict[str, object]) -> str:
        for key in ("item_id", "image_id", "id", "image_key"):
            value = item.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _format_score(self, value: object) -> str:
        try:
            return f"{float(value):.4f}"
        except Exception:
            return str(value) if value is not None else "-"

    def _build_image_get_target(
        self,
        item: Dict[str, object],
        settings: Dict[str, object],
    ) -> Tuple[str, Dict[str, str]]:
        item_id = self._extract_item_id(item)
        if not item_id:
            return "", {}

        template = str(settings["image_path_template"]).strip() or "/search_image/{item_id}"
        has_item_placeholder = "{item_id}" in template

        rendered_path = template
        rendered_path = rendered_path.replace("{item_id}", item_id)
        rendered_path = rendered_path.replace("{split}", str(item.get("split", "")).strip())
        rendered_path = rendered_path.replace(
            "{source_root_name}",
            str(item.get("source_root_name", "")).strip(),
        )

        params: Dict[str, str] = {}
        if not has_item_placeholder:
            query_key = str(settings.get("image_id_query_key", "item_id")).strip() or "item_id"
            params[query_key] = item_id

        return join_url(str(settings["api_base"]), rendered_path), params

    def _fetch_result_image(
        self,
        item: Dict[str, object],
        settings: Dict[str, object],
        headers: Dict[str, str],
    ) -> Tuple[object, str]:
        direct_ref = get_image_ref(item)
        if direct_ref:
            return direct_ref, ""

        url, params = self._build_image_get_target(item, settings)
        if not url:
            return None, "item_id를 찾을 수 없어 이미지 GET 요청을 만들 수 없습니다."

        try:
            resp = requests.get(
                url,
                params=params or None,
                timeout=int(settings["timeout_sec"]),
                verify=bool(settings["verify_ssl"]),
                headers=headers or None,
            )
            resp.raise_for_status()
        except Exception as exc:
            return None, f"image GET 실패: {exc}"

        content_type = str(resp.headers.get("content-type", "")).lower()
        if content_type.startswith("image/"):
            try:
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                return img, ""
            except Exception as exc:
                return None, f"이미지 디코딩 실패: {exc}"

        try:
            payload = resp.json()
        except Exception:
            return None, f"지원하지 않는 응답 타입: {content_type or 'unknown'}"

        if isinstance(payload, dict):
            ref = get_image_ref(payload)
            if ref:
                return ref, ""

        return None, "이미지 URL/경로를 응답에서 찾지 못했습니다."

    def _render_main_ui(self, settings: Dict[str, object], style_options: List[str]) -> None:
        st = self.st

        if not str(settings["api_base"]).strip():
            st.warning("`.env`의 `API_BASE_URL` 값이 비어 있습니다.")
        # st.caption(
        #     f"endpoint: {join_url(str(settings['api_base']), str(settings['search_path']))} | "
        #     f"image path: {str(settings['image_path_template'])}"
        # )

        st.subheader("User Input")
        uploaded = st.file_uploader("Upload user image", type=["jpg", "jpeg", "png"])
        col1, col2, col3 = st.columns(3)
        with col1:
            gender_filter = st.selectbox(
                "Gender",
                ["-- select gender --", "all", "male", "female"],
                index=0,
            )
        with col2:
            top_k = st.slider("Top-K", min_value=1, max_value=5, value=5, step=1)
        with col3:
            fallback_fill = st.checkbox("fallback_fill", value=True)

        # 선호/비선호 스타일은 서로 중복 선택되지 않도록 session_state를 정리합니다.
        preferred_key = "preferred_styles"
        disliked_key = "disliked_styles"
        gender_selected = gender_filter in {"all", "male", "female"}

        if not gender_selected:
            st.session_state[preferred_key] = []
            st.session_state[disliked_key] = []
            preferred_styles = st.multiselect(
                "Preferred styles",
                options=[],
                key=preferred_key,
                disabled=True,
                placeholder="Select gender first",
            )
            disliked_styles = st.multiselect(
                "Disliked styles",
                options=[],
                key=disliked_key,
                disabled=True,
                placeholder="Select gender first",
            )
        else:
            style_options_for_gender = self._style_options_for_gender(gender_filter, style_options)

            preferred_existing = [
                s for s in st.session_state.get(preferred_key, [])
                if s in style_options_for_gender
            ]
            disliked_existing = [
                s for s in st.session_state.get(disliked_key, [])
                if s in style_options_for_gender
            ]

            # 초기 충돌이 있으면 preferred를 우선 유지하고 disliked에서 제거합니다.
            disliked_existing = [s for s in disliked_existing if s not in preferred_existing]
            st.session_state[preferred_key] = preferred_existing
            st.session_state[disliked_key] = disliked_existing

            preferred_options = [s for s in style_options_for_gender if s not in disliked_existing]
            st.session_state[preferred_key] = [
                s for s in st.session_state[preferred_key] if s in preferred_options
            ]
            preferred_styles = st.multiselect(
                "Preferred styles",
                options=preferred_options,
                key=preferred_key,
            )

            disliked_options = [s for s in style_options_for_gender if s not in preferred_styles]
            st.session_state[disliked_key] = [
                s for s in st.session_state[disliked_key] if s in disliked_options
            ]
            disliked_styles = st.multiselect(
                "Disliked styles",
                options=disliked_options,
                key=disliked_key,
            )

        # st.subheader("Weight Controls")
        # beta_default = min(2.0, max(0.0, self._resolve_default_float("BETA_LIKE", 0.8)))
        # gamma_default = min(2.0, max(0.0, self._resolve_default_float("GAMMA_DISLIKE", 0.8)))
        # delta_default = min(2.0, max(0.0, self._resolve_default_float("DELTA_SURVEY_PRIOR", 1.0)))

        # beta_col, gamma_col, delta_col = st.columns(3)
        # with beta_col:
        #     beta_like = st.slider("like_weight", 0.0, 2.0, float(beta_default), 0.05)
        # with gamma_col:
        #     gamma_dislike = st.slider("weight_dislike", 0.0, 2.0, float(gamma_default), 0.05)
        # with delta_col:
        #     delta_survey_prior = st.slider("survey_prior_weight", 0.0, 2.0, float(delta_default), 0.05)

        beta_like, gamma_dislike, delta_survey_prior = 0.3, 0.3, 0.3

        if not st.button("Run search", type="primary"):
            return

        if not gender_selected:
            st.warning("Please select gender first.")
            return

        if uploaded is None:
            st.warning("Please upload an image.")
            return

        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", width=320)

        if settings["preview_mode"]:
            style_options_for_gender = self._style_options_for_gender(gender_filter, style_options)
            self._render_preview_results(
                top_k=int(top_k),
                gender_filter=gender_filter,
                preferred_styles=preferred_styles,
                style_options=style_options_for_gender,
            )
            return

        try:
            url, files, data, headers = self._build_request(
                uploaded=uploaded,
                gender_filter=gender_filter,
                top_k=int(top_k),
                preferred_styles=preferred_styles,
                disliked_styles=disliked_styles,
                fallback_fill=bool(fallback_fill),
                beta_like=float(beta_like),
                gamma_dislike=float(gamma_dislike),
                delta_survey_prior=float(delta_survey_prior),
                settings=settings,
            )
            with st.spinner("Calling backend /search ..."):
                resp = requests.post(
                    url,
                    files=files,
                    data=data,
                    timeout=int(settings["timeout_sec"]),
                    verify=bool(settings["verify_ssl"]),
                    headers=headers or None,
                )
                resp.raise_for_status()
                payload = resp.json()
        except Exception as exc:
            st.error(f"API request failed: {exc}")
            return

        rows = extract_results(payload)
        if len(rows) == 0:
            st.warning("Response contains no results.")
            st.json(payload)
            return

        normalized_rows: List[Dict[str, object]] = []
        for idx, row in enumerate(rows, start=1):
            rank = int(row.get("rank", idx))
            score = row.get("score", row.get("similarity", ""))
            style = row.get("style", "")
            gender = row.get("gender", "")
            item_id = self._extract_item_id(row)
            split = row.get("split", "")
            source_root_name = row.get("source_root_name", "")
            label = row.get("label", "")
            original_path = row.get("original_path", "")
            normalized_rows.append(
                {
                    "rank": rank,
                    "score": score,
                    "cosine_similarity": score,
                    "item_id": item_id,
                    "style": style,
                    "gender": gender,
                    "label": label,
                    "split": split,
                    "source_root_name": source_root_name,
                    "original_path": original_path,
                    "_raw_item": row,
                }
            )

        result_df = pd.DataFrame(
            [
                {
                    "rank": row["rank"],
                    "cosine_similarity": row["cosine_similarity"],
                    "item_id": row["item_id"],
                    "style": row["style"],
                    "gender": row["gender"],
                    "label": row["label"],
                    "split": row["split"],
                }
                for row in normalized_rows
            ]
        )
        # st.subheader("Top-K results")
        # st.dataframe(result_df, width="stretch")

        st.subheader("Recommendation gallery")
        gallery_cols = st.columns(3)
        for idx, item in enumerate(normalized_rows):
            col = gallery_cols[idx % 3]
            with col:
                image_obj, error_text = self._fetch_result_image(
                    item=item["_raw_item"],
                    settings=settings,
                    headers=headers,
                )
                if image_obj is not None:
                    if isinstance(image_obj, str):
                        if image_obj.startswith(("http://", "https://")):
                            st.image(image_obj, width="stretch")
                        else:
                            p = Path(image_obj)
                            if p.exists():
                                st.image(str(p), width="stretch")
                            else:
                                st.warning(f"이미지 경로를 찾지 못했습니다: {image_obj}")
                    else:
                        st.image(image_obj, width="stretch")
                else:
                    st.warning(error_text)

                st.markdown(f"**Rank #{item['rank']}**")
                st.caption(f"Cosine similarity: {self._format_score(item['score'])}")
                st.caption(
                    f"item_id={item['item_id']} | style={item['style']} | "
                    f"gender={item['gender']}"
                )

    def _render_preview_results(
        self,
        top_k: int,
        gender_filter: str,
        preferred_styles: Sequence[str],
        style_options: Sequence[str],
    ) -> None:
        st = self.st
        st.info("Preview mode: showing mock results without API call.")
        mock_rows: List[Dict[str, object]] = []

        for rank in range(1, top_k + 1):
            style = (
                preferred_styles[0]
                if len(preferred_styles) > 0
                else style_options[(rank - 1) % len(style_options)]
            )
            gender = gender_filter if gender_filter in {"male", "female"} else ("male" if rank % 2 else "female")
            mock_rows.append(
                {
                    "rank": rank,
                    "score": float(max(0.0, 1.0 - 0.03 * (rank - 1))),
                    "cosine_similarity": float(max(0.0, 1.0 - 0.03 * (rank - 1))),
                    "style": style,
                    "gender": gender,
                    "item_id": f"preview_{rank:03d}",
                }
            )

        st.subheader("Top-K results (preview)")
        st.dataframe(pd.DataFrame(mock_rows), width="stretch")


def is_running_in_streamlit() -> bool:
    try:
        import streamlit as st  # type: ignore

        return st.runtime.exists()  # type: ignore[attr-defined]
    except Exception:
        return False


def main() -> None:
    if is_running_in_streamlit():
        StreamlitApiRecommendationApp().run()
        return
    raise RuntimeError("This app is UI-only. Please run with: streamlit run app_api_ver.py")


if __name__ == "__main__":
    main()
