from pathlib import Path
import math
import re
import tempfile

import numpy as np
import pandas as pd
import streamlit as st


# ====================================================
# 1. 기본 설정
# ====================================================

st.set_page_config(
    page_title="Habitat-Prediction-Model",
    page_icon="🌿",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent

GROUPS = ["개구리", "매미", "새"]

HABITATS = [
    "도심/공원",
    "산림/숲",
    "초지/농경지",
    "습지/물가",
]

TIME_SLOTS = ["새벽", "오전", "한낮", "저녁", "밤"]

# 최고점과 10점 이내이면 함께 후보로 표시합니다.
# 검증된 생태학적 기준이 아닌 임시 표시 기준입니다.
CANDIDATE_MARGIN = 10.0


# ====================================================
# 2. 종별 생태 정보
#
# 기존에 제공한 수치를 사용합니다.
# 자료의 정확성은 별도로 확인해야 합니다.
#
# 서식 환경 코드 순서:
# 도심/공원, 산림/숲, 초지/농경지, 습지/물가
# ====================================================

SPECIES_DB = {
    "두꺼비": {
        "group": "개구리",
        "months": [2, 3, 4],
        "times": ["오전", "저녁", "밤"],
        "temp": [1.5, 12.3],
        "hab": [1, 1, 1, 1],
        "sound": "콕— 콕— 콕—",
        "desc": "갈색 몸에 돌기가 많고 눈 뒤에 독샘이 있다. 봄철 물가에서 번식한다.",
    },
    "청개구리": {
        "group": "개구리",
        "months": [4, 5, 6],
        "times": ["오전", "저녁", "밤"],
        "temp": [12.3, 21.6],
        "hab": [1, 1, 1, 1],
        "sound": "꽥 꽥 꽥 / 깩 깩 깩",
        "desc": "발가락 끝에 흡반이 있어 풀과 나무를 잘 타고 논과 웅덩이에서 번식한다.",
    },
    "참개구리": {
        "group": "개구리",
        "months": [4, 5, 6],
        "times": ["저녁", "밤"],
        "temp": [12.3, 21.6],
        "hab": [1, 0, 1, 1],
        "sound": "까르르르르 / 꾸르륵 꾸르륵",
        "desc": "등에 검은 반점과 밝은 중앙 줄이 있고 논과 연못 등에서 생활한다.",
    },
    "황소개구리": {
        "group": "개구리",
        "months": [4, 5, 6, 7],
        "times": ["저녁", "밤"],
        "temp": [12.3, 24.8],
        "hab": [1, 0, 1, 1],
        "sound": "우음— 우음— / 우우웅",
        "desc": "몸집이 크고 눈 뒤 고막이 뚜렷하다. 저수지와 연못 등에서 생활한다.",
    },
    "북방산개구리": {
        "group": "개구리",
        "months": [2, 3, 4],
        "times": ["한낮", "저녁", "밤"],
        "temp": [1.5, 12.3],
        "hab": [1, 1, 1, 1],
        "sound": "호르르릉 / 오그그그그",
        "desc": "산림에서 생활하고 이른 봄 논이나 계곡 주변 물웅덩이에서 번식한다.",
    },
    "맹꽁이": {
        "group": "개구리",
        "months": [5, 6, 7, 8],
        "times": ["저녁", "밤"],
        "temp": [17.5, 25.3],
        "hab": [1, 1, 1, 1],
        "sound": "맹— 맹— / 꽁— 꽁—",
        "desc": "몸통이 둥글고 팔다리가 짧다. 장마철 물웅덩이에서 번식하며 비가 오면 낮에도 운다.",
    },
    "유지매미": {
        "group": "매미",
        "months": [6, 7, 8, 9],
        "times": ["오전", "한낮", "저녁"],
        "temp": [20.4, 28.2],
        "hab": [1, 1, 1, 0],
        "sound": "찍 끄르르르 / 딕! 쯔그르르르",
        "desc": "몸길이 34~36mm, 날개 끝까지 약 50~60mm.",
    },
    "애매미": {
        "group": "매미",
        "months": [6, 7, 8, 9],
        "times": ["오전", "한낮", "저녁"],
        "temp": [20.4, 28.2],
        "hab": [1, 1, 1, 0],
        "sound": "스삐오스 스삐오스 치르르르",
        "desc": "작은 크기를 가지며 해발 600m 이하에서 주로 서식한다.",
    },
    "참매미": {
        "group": "매미",
        "months": [6, 7, 8, 9],
        "times": ["새벽", "오전", "한낮", "저녁"],
        "temp": [20.4, 28.2],
        "hab": [1, 1, 1, 0],
        "sound": "맴맴맴",
        "desc": "머리와 가슴이 단단하고 배는 짧고 두툼하다.",
    },
    "쓰름매미": {
        "group": "매미",
        "months": [7, 8],
        "times": ["오전", "저녁"],
        "temp": [21.4, 29.9],
        "hab": [1, 1, 1, 0],
        "sound": "쓰름쓰름",
        "desc": "울음소리가 쓰름, 쓰름으로 들려 이름이 붙었다.",
    },
    "털매미": {
        "group": "매미",
        "months": [6, 7, 8, 9],
        "times": ["오전", "한낮", "저녁"],
        "temp": [15.0, 34.0],
        "hab": [1, 1, 1, 0],
        "sound": "찌이이이— / 쓰으으으—",
        "desc": "몸길이 약 20~24mm, 날개 끝까지 약 35~40mm. 길게 울다가 음높이가 바뀐다.",
    },
    "말매미": {
        "group": "매미",
        "months": [6, 7, 8, 9, 10],
        "times": ["오전", "한낮", "밤"],
        "temp": [24.0, 30.0],
        "hab": [1, 1, 1, 0],
        "sound": "찌르르르",
        "desc": "우리나라 매미 중 가장 큰 종으로 한반도 전역에 서식한다.",
    },
    "물까치": {
        "group": "새",
        "months": list(range(1, 13)),
        "times": ["오전", "한낮", "저녁"],
        "temp": [-4.6, 24.9],
        "hab": [1, 1, 1, 0],
        "sound": "꺄악-꺅꺅꺅꺅꺅",
        "desc": "일반적으로 무리 지어 생활하는 텃새로 한반도 전역에 서식한다.",
    },
    "까마귀": {
        "group": "새",
        "months": list(range(1, 13)),
        "times": ["오전", "한낮", "저녁"],
        "temp": [-0.9, 25.1],
        "hab": [1, 1, 1, 1],
        "sound": "까악— 까악—",
        "desc": "산림·농경지·공원·하천 등에서 연중 생활하는 검은색 새이다.",
    },
    "멧비둘기": {
        "group": "새",
        "months": list(range(1, 13)),
        "times": ["오전", "한낮"],
        "temp": [-15.0, 35.0],
        "hab": [1, 1, 0, 0],
        "sound": "구, 구, 쿠-, 쿠-",
        "desc": "몸길이 약 33cm. 목 옆에 회색과 검은색 줄무늬 반점이 있다.",
    },
    "뻐꾸기": {
        "group": "새",
        "months": [4, 5, 6, 7, 8, 9],
        "times": ["오전", "한낮", "저녁", "밤"],
        "temp": [12.3, 25.3],
        "hab": [1, 1, 1, 1],
        "sound": "뻐꾹- / 삐삐삐삐",
        "desc": "몸길이 32~36cm. 눈과 눈테가 노란색이고 몸은 옅은 검은색이다.",
    },
    "까치": {
        "group": "새",
        "months": list(range(1, 13)),
        "times": ["오전", "한낮"],
        "temp": [-0.6, 25.3],
        "hab": [1, 1, 1, 0],
        "sound": "꺅-꺅-꺅-꺅",
        "desc": "윗면은 푸른 광택의 검은색이고 배는 흰색이다.",
    },
    "참새": {
        "group": "새",
        "months": list(range(1, 13)),
        "times": ["오전", "한낮", "저녁"],
        "temp": [25.0, 35.0],
        "hab": [1, 0, 1, 0],
        "sound": "짹 짹",
        "desc": "윗면은 밤색이고 등에 검은 줄무늬가 있으며 가슴과 배는 흰색이다.",
    },
}


# ====================================================
# 3. 생태환경 계산
# ====================================================

def calculate_environment(values):
    for species, value in values.items():
        if (
            species not in SPECIES_DB
            or not math.isfinite(value)
            or not 0 <= value <= 100
        ):
            raise ValueError("종별 점수는 0~100 사이여야 합니다.")

    positive = {
        species: float(value)
        for species, value in values.items()
        if value > 0
    }

    total = sum(positive.values())

    if total <= 0:
        raise ValueError("생물 종 점수를 하나 이상 입력해 주세요.")

    weights = sorted(
        [
            (species, value / total)
            for species, value in positive.items()
        ],
        key=lambda item: item[1],
        reverse=True,
    )

    month_scores = [
        sum(
            weight * 100
            for species, weight in weights
            if month in SPECIES_DB[species]["months"]
        )
        for month in range(1, 13)
    ]

    time_scores = [
        sum(
            weight * 100
            for species, weight in weights
            if slot in SPECIES_DB[species]["times"]
        )
        for slot in TIME_SLOTS
    ]

    low_temp = sum(
        weight * SPECIES_DB[species]["temp"][0]
        for species, weight in weights
    )

    high_temp = sum(
        weight * SPECIES_DB[species]["temp"][1]
        for species, weight in weights
    )

    # 각 환경의 점수는 각각 0~100입니다.
    # 환경 점수의 합계를 100으로 맞추지 않습니다.
    habitat_scores = [
        sum(
            weight * SPECIES_DB[species]["hab"][index] * 100
            for species, weight in weights
        )
        for index in range(len(HABITATS))
    ]

    highest = max(habitat_scores)

    candidates = [
        name
        for name, score in zip(HABITATS, habitat_scores)
        if (
            highest > 0
            and score > 0
            and highest - score <= CANDIDATE_MARGIN + 1e-9
        )
    ]

    if not candidates:
        judgment = "등록된 자료로 판단할 수 없습니다."

    elif len(candidates) == 1:
        judgment = (
            "현재 입력에서 상대적으로 지지되는 환경입니다. "
            "실제 장소로 확정할 수는 없습니다."
        )

    else:
        judgment = (
            "여러 환경이 비슷하게 지지됩니다. "
            "현재 종 정보만으로 구분하기 어렵습니다."
        )

    return {
        "raw": positive,
        "weights": weights,
        "months": month_scores,
        "times": time_scores,
        "low": low_temp,
        "high": high_temp,
        "habitat_scores": habitat_scores,
        "candidates": candidates,
        "judgment": judgment,
        "shannon": -sum(
            weight * math.log(weight)
            for _, weight in weights
        ),
        "simpson": sum(
            weight ** 2
            for _, weight in weights
        ),
    }


# ====================================================
# 4. 입력값 초기화 / 예시 / 붙여넣기
# ====================================================

def reset_inputs():
    for species in SPECIES_DB:
        st.session_state["v3_score_" + species] = 0.0

    for group in GROUPS:
        st.session_state["v3_bg_" + group] = 0.0

    st.session_state["v3_paste"] = ""
    st.session_state.pop("v3_saved_input", None)
    st.session_state.pop("v3_paste_message", None)


def load_example():
    reset_inputs()

    st.session_state["v3_score_털매미"] = 99.89
    st.session_state["v3_score_애매미"] = 0.02
    st.session_state["v3_bg_매미"] = 0.09


def apply_paste():
    text = st.session_state.get("v3_paste", "")
    parsed = {}

    for species in SPECIES_DB:
        pattern = (
            re.escape(species)
            + r"\s+([+-]?\d+(?:\.\d+)?)"
        )

        match = re.search(pattern, text)

        if match:
            value = float(match.group(1))

            if not math.isfinite(value) or not 0 <= value <= 100:
                st.session_state["v3_paste_message"] = (
                    False,
                    f"{species}: 점수는 0~100 사이여야 합니다.",
                )
                return

            parsed[species] = value

    if not parsed:
        st.session_state["v3_paste_message"] = (
            False,
            "종 이름 뒤에 점수를 입력하세요. 예: 털매미 99.89%",
        )
        return

    for species in SPECIES_DB:
        st.session_state["v3_score_" + species] = float(
            parsed.get(species, 0)
        )

    for group in GROUPS:
        st.session_state["v3_bg_" + group] = 0.0

    st.session_state.pop("v3_saved_input", None)

    st.session_state["v3_paste_message"] = (
        True,
        f"{len(parsed)}종의 점수를 반영했습니다.",
    )


# ====================================================
# 5. 모델 파일 찾기
# ====================================================

def find_model_files():
    folders = [
        BASE_DIR / "models" / "cicada",
        BASE_DIR,
    ]

    names = [
        "model.keras",
        "걸생 베타 매미 종 동정 모델.keras",
    ]

    for folder in folders:
        for name in names:
            paths = (
                folder / name,
                folder / "classes.npy",
                folder / "feature_config.npz",
            )

            if all(path.is_file() for path in paths):
                return paths

    return None


@st.cache_resource(show_spinner=False)
def load_model_files(
    model_path,
    classes_path,
    config_path,
    file_version,
):
    import tensorflow as tf

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    categories = np.load(
        classes_path,
        allow_pickle=False,
    )

    with np.load(config_path, allow_pickle=False) as config:
        sample_rate = int(config["sample_rate"].item())
        n_mfcc = int(config["n_mfcc"].item())

    if categories.ndim != 1 or len(categories) < 2:
        raise ValueError("카테고리 목록이 올바르지 않습니다.")

    categories = [str(category) for category in categories]

    if len(set(categories)) != len(categories):
        raise ValueError("카테고리 이름이 중복됩니다.")

    if sample_rate <= 0 or n_mfcc <= 0:
        raise ValueError("특징 추출 설정이 올바르지 않습니다.")

    if model.input_shape[-1] != n_mfcc:
        raise ValueError("모델과 특징 추출 설정이 일치하지 않습니다.")

    if model.output_shape[-1] != len(categories):
        raise ValueError("모델과 카테고리 수가 일치하지 않습니다.")

    allowed = {
        species
        for species, info in SPECIES_DB.items()
        if info["group"] == "매미"
    } | {"배경소리"}

    if not set(categories).issubset(allowed):
        raise ValueError("매미 분류 모델을 연결해 주세요.")

    return model, categories, sample_rate, n_mfcc


# ====================================================
# 6. 음원 분류
# ====================================================

def classify_audio(uploaded, paths):
    import librosa

    suffix = Path(uploaded.name).suffix.lower()

    if suffix not in {".mp3", ".wav"}:
        raise ValueError("MP3 또는 WAV 파일을 사용해 주세요.")

    if uploaded.size > 20 * 1024 * 1024:
        raise ValueError("20MB 이하 파일을 사용해 주세요.")

    version = tuple(
        (path.stat().st_mtime_ns, path.stat().st_size)
        for path in paths
    )

    model, categories, sample_rate, n_mfcc = load_model_files(
        str(paths[0]),
        str(paths[1]),
        str(paths[2]),
        version,
    )

    with tempfile.TemporaryDirectory() as folder:
        audio_path = Path(folder) / ("audio" + suffix)
        audio_path.write_bytes(uploaded.getvalue())

        audio, sr = librosa.load(
            str(audio_path),
            sr=sample_rate,
            mono=True,
            duration=61,
        )

    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise ValueError("음원이 비어 있거나 잘못된 값이 있습니다.")

    duration = len(audio) / sr

    if duration > 60:
        raise ValueError("60초 이하 음원을 사용해 주세요.")

    if np.max(np.abs(audio)) == 0:
        raise ValueError("완전히 무음인 파일은 분석할 수 없습니다.")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
    )

    feature = np.mean(mfcc, axis=1).astype(np.float32)

    if not np.all(np.isfinite(feature)):
        raise ValueError("특징 추출에 실패했습니다.")

    scores = np.asarray(
        model.predict(
            feature.reshape(1, -1),
            verbose=0,
        )[0],
        dtype=float,
    )

    if (
        scores.shape != (len(categories),)
        or not np.all(np.isfinite(scores))
        or np.any(scores < 0)
        or np.any(scores > 1)
        or not np.isclose(scores.sum(), 1, atol=1e-3)
    ):
        raise ValueError("모델 예측 결과가 올바르지 않습니다.")

    return {
        category: float(score * 100)
        for category, score in zip(categories, scores)
    }, duration


# ====================================================
# 7. 결과 화면
# ====================================================

def show_result(result):
    st.divider()
    st.subheader("생태환경 참고 결과")

    # 서식 환경을 가장 먼저 표시
    st.markdown("### 서식 환경 후보")

    if result["candidates"]:
        st.write(" · ".join(result["candidates"]))
    else:
        st.write("판단 불가")

    st.info(result["judgment"])

    habitat_table = pd.DataFrame({
        "서식 환경": HABITATS,
        "근거 점수 (0~100)": [
            round(score, 1)
            for score in result["habitat_scores"]
        ],
        "후보 여부": [
            "후보" if name in result["candidates"] else "—"
            for name in HABITATS
        ],
    }).sort_values(
        "근거 점수 (0~100)",
        ascending=False,
    )

    st.dataframe(
        habitat_table,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "점수는 실제 서식지의 확률이 아닙니다. "
        "환경별 점수를 합계 100으로 환산하지 않습니다. "
        "100점도 실제 장소라는 확정을 뜻하지 않습니다."
    )

    with st.expander("서식 환경 계산 설명"):
        st.write(
            "종별 입력 점수를 합계 1로 환산한 뒤, "
            "등록된 서식 환경 코드와 곱해 더합니다."
        )

        st.write(
            f"최고점과 {CANDIDATE_MARGIN:.0f}점 이내인 환경을 "
            "함께 후보로 표시합니다."
        )

        st.caption(
            "10점 차이는 검증된 생태학적 기준이 아닌 "
            "임시 표시 기준입니다."
        )

        st.warning(
            "현재 매미 6종은 서식 환경 코드가 같습니다. "
            "따라서 매미 종 정보만으로 도심·산림·농경지를 "
            "구분할 수 없습니다."
        )

    st.markdown("### 시기와 활동 조건")

    best_months = [
        f"{index + 1}월"
        for index, score in enumerate(result["months"])
        if abs(score - max(result["months"])) < 0.01
    ]

    best_times = [
        slot
        for slot, score in zip(TIME_SLOTS, result["times"])
        if abs(score - max(result["times"])) < 0.01
    ]

    columns = st.columns(3)

    with columns[0]:
        st.caption("참고 시기")
        st.write(" · ".join(best_months))

    with columns[1]:
        st.caption("활동 시간")
        st.write(" · ".join(best_times))

    with columns[2]:
        st.caption("참고 기온 범위")
        st.write(
            f'{result["low"]:.1f} ~ {result["high"]:.1f} °C'
        )

    dominant, weight = result["weights"][0]

    st.success(
        f"가장 크게 반영된 종: {dominant}"
        f" · 반영 비중 {weight * 100:.1f}%"
    )

    st.caption(
        "시기·시간·기온도 등록된 생태 정보를 조합한 참고값이며 "
        "실제 녹음 환경의 관측값이나 확률이 아닙니다."
    )

    with st.expander("종별 비중과 상세 계산"):
        st.dataframe(
            pd.DataFrame([
                {
                    "종": species,
                    "그룹": SPECIES_DB[species]["group"],
                    "입력 점수 (%)": round(
                        result["raw"][species], 2
                    ),
                    "반영 비중 (%)": round(
                        weight * 100, 2
                    ),
                }
                for species, weight in result["weights"]
            ]),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("**월별 참고 적합도**")

        st.dataframe(
            pd.DataFrame({
                "월": [f"{month}월" for month in range(1, 13)],
                "적합도 (%)": result["months"],
            }),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("**시간대별 참고 적합도**")

        st.dataframe(
            pd.DataFrame({
                "시간대": TIME_SLOTS,
                "적합도 (%)": result["times"],
            }),
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            f'점수 기반 H′: {result["shannon"]:.4f}'
            f' / 점수 기반 D: {result["simpson"]:.4f}'
        )

        st.info(
            "H′와 D는 분류 점수의 분포를 계산한 값입니다. "
            "실제 개체수 기반 생물다양성 지수가 아닙니다. "
            "서로 다른 모델의 점수가 같은 기준으로 "
            "비교 가능하다는 보장도 없습니다."
        )

    with st.expander("종별 생태 사전"):
        for species, _ in result["weights"]:
            info = SPECIES_DB[species]

            st.markdown(
                f'**{species} · {info["group"]}**'
            )

            st.write(
                "출현 월: "
                + ", ".join(map(str, info["months"]))
                + "월"
            )

            st.write(
                "활동 시간: "
                + " · ".join(info["times"])
            )

            st.write(
                f'참고 기온: {info["temp"][0]}'
                f' ~ {info["temp"][1]} °C'
            )

            habitat_names = [
                name
                for name, flag in zip(
                    HABITATS,
                    info["hab"],
                )
                if flag
            ]

            st.write(
                "등록된 서식 환경: "
                + ", ".join(habitat_names)
            )

            st.write("울음소리: " + info["sound"])
            st.write(info["desc"])
            st.divider()


# ====================================================
# 8. 수동 입력 화면
# ====================================================

def manual_page():
    for species in SPECIES_DB:
        st.session_state.setdefault(
            "v3_score_" + species,
            0.0,
        )

    for group in GROUPS:
        st.session_state.setdefault(
            "v3_bg_" + group,
            0.0,
        )

    left, right = st.columns(2)

    with left:
        st.button(
            "예시 넣기",
            on_click=load_example,
            use_container_width=True,
        )

    with right:
        st.button(
            "초기화",
            on_click=reset_inputs,
            use_container_width=True,
        )

    with st.expander("모델 결과 텍스트 붙여넣기"):
        st.text_area(
            "종 이름과 점수",
            placeholder=(
                "1 털매미 99.89%\n"
                "2 배경소리 0.09%\n"
                "3 애매미 0.02%"
            ),
            key="v3_paste",
        )

        st.button(
            "점수 반영",
            on_click=apply_paste,
        )

        st.caption(
            "생물 종 점수만 반영합니다. "
            "배경소리는 아래에서 직접 입력하세요."
        )

        message = st.session_state.get("v3_paste_message")

        if message:
            success, text = message

            if success:
                st.success(text)
            else:
                st.error(text)

    with st.form("v3_manual_form"):
        columns = st.columns(3)

        for column, group in zip(columns, GROUPS):
            with column:
                st.markdown(f"### {group}")

                for species, info in SPECIES_DB.items():
                    if info["group"] != group:
                        continue

                    st.number_input(
                        f"{species} (%)",
                        min_value=0.0,
                        max_value=100.0,
                        step=0.01,
                        format="%.2f",
                        key="v3_score_" + species,
                    )

                st.number_input(
                    f"{group} 배경소리 (%)",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.01,
                    format="%.2f",
                    key="v3_bg_" + group,
                )

        submitted = st.form_submit_button(
            "생태환경 추정하기",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        values = {
            species: float(
                st.session_state["v3_score_" + species]
            )
            for species in SPECIES_DB
        }

        backgrounds = {
            group: float(
                st.session_state["v3_bg_" + group]
            )
            for group in GROUPS
        }

        try:
            calculate_environment(values)

            st.session_state["v3_saved_input"] = {
                "values": values,
                "backgrounds": backgrounds,
            }

        except ValueError as error:
            st.session_state.pop("v3_saved_input", None)
            st.error(str(error))

    saved = st.session_state.get("v3_saved_input")

    if saved:
        if any(
            value > 0
            for value in saved["backgrounds"].values()
        ):
            st.caption(
                "배경소리를 제외한 생물 종 점수만 사용합니다. "
                "배경 점수가 높다면 실제 종 검출 여부를 "
                "먼저 확인해 주세요."
            )

        show_result(
            calculate_environment(saved["values"])
        )


# ====================================================
# 9. 음원 자동 분석 화면
# ====================================================

def audio_page():
    st.subheader("매미 녹음 파일 분석")

    st.caption(
        "MP3·WAV / 최대 20MB / 60초 이하. "
        "파일 전체의 대표 소리를 분류합니다."
    )

    paths = find_model_files()

    if paths is None:
        st.info(
            "매미 모델 파일이 아직 없습니다. "
            "'점수 직접 입력' 기능은 바로 사용할 수 있습니다."
        )

        with st.expander("모델 파일 넣는 위치"):
            st.code(
                "models/\n"
                "└── cicada/\n"
                "    ├── model.keras\n"
                "    ├── classes.npy\n"
                "    └── feature_config.npz",
                language="text",
            )

            st.caption(
                "기존 모델명 "
                "'걸생 베타 매미 종 동정 모델.keras'도 지원합니다. "
                "세 파일은 같은 학습에서 생성된 것이어야 합니다. "
                "개구리·새 자동 분류는 연결되어 있지 않습니다."
            )

    uploaded = st.file_uploader(
        "녹음 파일 선택",
        type=["mp3", "wav"],
        key="v3_audio",
    )

    if uploaded is None:
        return

    if uploaded.size > 20 * 1024 * 1024:
        st.error("20MB 이하 파일을 사용해 주세요.")
        return

    st.audio(uploaded)

    if st.button(
        "음원 분석하기",
        type="primary",
        disabled=paths is None,
    ):
        try:
            with st.spinner("소리를 분석하고 있습니다…"):
                scores, duration = classify_audio(
                    uploaded,
                    paths,
                )

            st.caption(f"음원 길이: {duration:.1f}초")

            st.dataframe(
                pd.DataFrame(
                    sorted(
                        scores.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    ),
                    columns=["종", "모델 점수 (%)"],
                ),
                hide_index=True,
                use_container_width=True,
            )

            st.warning(
                "모델 점수는 정답 확률이나 실제 정확도가 아닙니다. "
                "학습하지 않은 소리도 등록된 종류로 분류될 수 있으며, "
                "여러 종을 개별 검출하는 방식은 아닙니다."
            )

            best_species = max(scores, key=scores.get)

            if best_species == "배경소리":
                st.info(
                    "배경소리 점수가 가장 높아 환경 추정을 보류했습니다."
                )
                return

            values = {
                species: score
                for species, score in scores.items()
                if species in SPECIES_DB
            }

            st.caption(
                "배경소리를 제외한 생물 종 점수를 "
                "합계 100%로 환산해 반영합니다."
            )

            show_result(
                calculate_environment(values)
            )

        except Exception as error:
            st.error(f"분석하지 못했습니다: {error}")


# ====================================================
# 10. 메인 화면
# ====================================================

def main():
    st.title("소리로 살펴보는 생태환경")

    # 새 코드가 적용됐는지 확인하는 표시입니다.
    st.success("서식지 후보 방식 · V3")

    st.caption(
        "종별 예측 점수를 입력하거나 매미 녹음 파일을 올려 보세요."
    )

    manual_tab, audio_tab = st.tabs([
        "점수 직접 입력",
        "음원으로 자동 분석",
    ])

    with manual_tab:
        manual_page()

    with audio_tab:
        audio_page()

    st.divider()

    st.caption(
        "Habitat-Prediction-Model · "
        "생태 자료는 사용자가 제공한 기존 값이며 "
        "별도 검증이 필요합니다."
    )

    st.caption(
        "온라인 배포 시 음원은 서버로 전송되어 처리됩니다. "
        "이 앱 코드는 업로드 음원을 영구 저장하지 않습니다."
    )


if __name__ == "__main__":
    main()
