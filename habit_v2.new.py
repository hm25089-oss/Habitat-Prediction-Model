from pathlib import Path
import io
import math
import re

import numpy as np
import pandas as pd
import streamlit as st


# =====================================================
# 1. 기본 설정
# =====================================================

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

# 최고점과 이 점수 이내인 환경을 함께 표시합니다.
# 검증된 생태 기준이 아니라 화면 표시용 임시 기준입니다.
CANDIDATE_MARGIN = 10.0

PREFIX = "habitat_v4_"


# =====================================================
# 2. 종별 정보
#
# hab: 0~10 잠정 서식지 적합도
# 순서: 도심/공원, 산림/숲, 초지/농경지, 습지/물가
#
# months: 사용자가 제공한 관찰·울음 시기
# temp: 사용자가 제공한 참고 범위
#       실제 생존·활동 한계가 아닙니다.
# conditional_times: 조건이 충족될 때 추가되는 시간
# =====================================================

SPECIES_DB = {
    "두꺼비": {
        "group": "개구리",
        "months": [2, 3, 4],
        "times": ["저녁", "밤"],
        "conditional_times": ["오전", "한낮"],
        "condition": "번식기·비 오는 날에는 낮에도 관찰될 수 있음",
        "temp": [1.5, 12.3],
        "hab": [0, 10, 5, 10],
        "sound": "콕— 콕— 콕—",
        "desc": (
            "갈색 몸에 울퉁불퉁한 돌기가 많고 눈 뒤에 독샘이 있다. "
            "큰 개체는 약 15cm에 이른다. 평소 육상에서 생활하다 "
            "봄철 물가에 모여 두 줄의 긴 알주머니를 낳는다."
        ),
    },
    "청개구리": {
        "group": "개구리",
        "months": [4, 5, 6],
        "times": ["저녁", "밤"],
        "conditional_times": ["오전", "한낮"],
        "condition": "비 오거나 흐린 날에는 낮에도 울 수 있음",
        "temp": [12.3, 21.6],
        "hab": [0, 5, 10, 10],
        "sound": "꽥 꽥 꽥 / 깩 깩 깩",
        "desc": (
            "몸길이 약 2.5~4cm로 발가락 끝에 흡반이 있어 "
            "풀과 나무를 잘 탄다. 등은 주로 녹색이나 "
            "주변 환경에 따라 회갈색 등으로 변하며 "
            "논과 웅덩이에서 번식한다."
        ),
    },
    "참개구리": {
        "group": "개구리",
        "months": [4, 5, 6],
        "times": ["저녁", "밤"],
        "temp": [12.3, 21.6],
        "hab": [0, 0, 10, 10],
        "sound": "까르르르르 / 꾸르륵 꾸르륵",
        "desc": (
            "몸길이 약 6~9cm로 녹색이나 갈색 등에 검은 반점이 있다. "
            "등 중앙의 밝은 줄과 양옆의 융기선이 특징이며 "
            "논과 연못 등 물가에서 생활한다."
        ),
    },
    "황소개구리": {
        "group": "개구리",
        "months": [4, 5, 6, 7],
        "times": ["저녁", "밤"],
        "temp": [12.3, 24.8],
        "hab": [0, 0, 0, 10],
        "sound": "우음— 우음— / 우우웅 우우웅",
        "desc": (
            "몸길이 약 15~20cm로 몸집이 크고 눈 뒤 고막이 뚜렷하다. "
            "녹색이나 갈색을 띠며 저수지와 연못에서 "
            "물고기·곤충·다른 개구리 등을 잡아먹는다."
        ),
        "note": (
            "요청한 서식지 점수를 그대로 입력했다. "
            "습지/물가 1점은 저수지·연못 생활 설명과 "
            "불일치할 수 있으므로 재검토가 필요하다."
        ),
    },
    "북방산개구리": {
        "group": "개구리",
        "months": [2, 3, 4],
        "times": ["오전", "한낮", "저녁", "밤"],
        "temp": [1.5, 12.3],
        "hab": [0, 10, 5, 10],
        "sound": "호르르릉 호르릉 / 오그그그그",
        "desc": (
            "몸길이 약 5~8.5cm로 등은 황갈색이나 적갈색이며 "
            "눈 뒤에 짙은 무늬가 있다. 산림에서 생활하고 "
            "이른 봄 논이나 계곡 주변 물웅덩이에서 번식한다."
        ),
        "note": "활동 시간 정보는 번식기 기준이다.",
    },
    "맹꽁이": {
        "group": "개구리",
        "months": [5, 6, 7, 8],
        "times": ["저녁", "밤"],
        "conditional_times": ["오전", "한낮"],
        "condition": "번식기에 비가 오거나 흐리면 낮에도 울 수 있음",
        "temp": [17.5, 25.3],
        "hab": [5, 0, 5, 10],
        "sound": "맹— 맹— / 꽁— 꽁—",
        "desc": (
            "몸길이 약 3.5~5.5cm로 몸통이 둥글고 "
            "머리와 팔다리가 짧다. 암갈색 등에 작은 반점과 돌기가 있다. "
            "평소 땅속에 숨어 지내다가 장마철 얕은 물웅덩이에 모여 번식한다."
        ),
        "note": (
            "출현 월은 울음으로 관찰하기 좋은 기간 기준이다. "
            "요청한 습지/물가 1점은 번식지 설명과 "
            "불일치할 수 있어 재검토가 필요하다."
        ),
    },
    "유지매미": {
        "group": "매미",
        "months": [6, 7, 8, 9],
        "times": ["오전", "한낮", "저녁"],
        "temp": [20.4, 28.2],
        "hab": [5, 5, 0, 0],
        "sound": "찍 끄르르르 / 딕! 쯔그르르르",
        "desc": (
            "몸길이는 약 34~36mm이며 "
            "날개 끝까지의 길이는 약 50~60mm이다."
        ),
    },
    "애매미": {
        "group": "매미",
        "months": [6, 7, 8, 9],
        "times": ["오전", "한낮", "저녁"],
        "temp": [20.4, 28.2],
        "hab": [5, 10, 0, 0],
        "sound": "스삐오스 스삐오스 치르르르",
        "desc": "작은 크기를 가지며 해발 600m 이하에서 주로 서식한다.",
    },
    "참매미": {
        "group": "매미",
        "months": [6, 7, 8, 9],
        "times": ["새벽", "오전", "한낮", "저녁"],
        "temp": [20.4, 28.2],
        "hab": [10, 5, 0, 0],
        "sound": "맴맴맴",
        "desc": (
            "머리와 가슴이 단단하고 배는 짧고 두툼해 "
            "날개를 접고 앉으면 둥글고 안정적인 인상을 준다."
        ),
    },
    "쓰름매미": {
        "group": "매미",
        "months": [7, 8],
        "times": ["오전", "저녁"],
        "temp": [21.4, 29.9],
        "hab": [5, 5, 0, 0],
        "sound": "쓰름쓰름",
        "desc": (
            "울음소리가 '쓰름, 쓰름'으로 들린다 하여 "
            "쓰름매미라는 이름이 붙여졌다."
        ),
    },
    "털매미": {
        "group": "매미",
        "months": [6, 7, 8, 9],
        "times": ["오전", "한낮", "저녁"],
        "temp": [15.0, 34.0],
        "hab": [5, 5, 0, 0],
        "sound": "찌이이이이— / 쓰으으으—",
        "desc": (
            "몸길이 약 20~24mm, 날개 끝까지 약 35~40mm이다. "
            "길게 이어지는 울음 도중 음높이가 바뀐다."
        ),
    },
    "말매미": {
        "group": "매미",
        "months": [6, 7, 8, 9, 10],
        "times": ["오전", "한낮", "밤"],
        "temp": [24.0, 30.0],
        "hab": [10, 5, 0, 0],
        "sound": "찌르르르",
        "desc": (
            "우리나라 매미 중 가장 큰 종으로 "
            "한반도 전역에 서식한다."
        ),
    },
    "물까치": {
        "group": "새",
        "months": list(range(1, 13)),
        "times": ["오전", "한낮", "저녁"],
        "temp": [-4.6, 24.9],
        "hab": [10, 10, 0, 0],
        "sound": "꺄악-꺅꺅꺅꺅꺅",
        "desc": (
            "일반적으로 무리 지어 생활하는 텃새이다. "
            "분포 설명은 지역 자료로 별도 확인할 필요가 있다."
        ),
    },
    "까마귀": {
        "group": "새",
        "months": list(range(1, 13)),
        "times": ["오전", "한낮", "저녁"],
        "temp": [-0.9, 25.1],
        "hab": [0, 10, 5, 5],
        "sound": "까악— 까악— 까악—",
        "desc": (
            "몸길이 약 50cm로 온몸과 부리·다리가 검다. "
            "산림·농경지·공원·하천 주변에서 생활하며 "
            "곤충·곡류·열매 등을 먹는 잡식성 새이다."
        ),
        "note": (
            "이 사전의 '까마귀' 항목을 큰부리까마귀·송장까마귀의 "
            "합쳐진 분류 결과에 적용할 경우 종별 생태 차이가 생략된다."
        ),
    },
    "멧비둘기": {
        "group": "새",
        "months": list(range(1, 13)),
        "times": ["오전", "한낮"],
        "temp": [-15.0, 35.0],
        "hab": [5, 10, 10, 0],
        "sound": "구, 구, 쿠-, 쿠-",
        "desc": (
            "몸길이 약 33cm로 머리·목·몸 아랫면은 회갈색이다. "
            "목 옆에 회색과 검은색 줄무늬 반점이 있다. "
            "울음의 뒤 두 음절은 더 높고 빠르다."
        ),
        "note": "사용자가 제공한 번식 시기: 2~7월.",
    },
    "뻐꾸기": {
        "group": "새",
        "months": [4, 5, 6, 7, 8, 9],
        "times": ["오전", "한낮", "저녁", "밤"],
        "temp": [12.3, 25.3],
        "hab": [0, 10, 5, 0],
        "sound": "뻐꾹- / 삐삐삐삐",
        "desc": (
            "몸길이 약 32~36cm이며 "
            "눈과 눈테가 노란색이고 몸은 옅은 검은색이다."
        ),
    },
    "까치": {
        "group": "새",
        "months": list(range(1, 13)),
        "times": ["오전", "한낮"],
        "temp": [-0.6, 25.3],
        "hab": [10, 0, 10, 0],
        "sound": "꺅-꺅-꺅-꺅",
        "desc": (
            "몸 전체 길이는 약 46cm로 "
            "윗면은 푸른 광택이 있는 검은색이고 배는 흰색이다."
        ),
    },
    "참새": {
        "group": "새",
        "months": list(range(1, 13)),
        "times": ["오전", "한낮", "저녁"],
        "temp": None,
        "hab": [10, 0, 10, 5],
        "sound": "짹 짹",
        "desc": (
            "몸길이 약 14.5cm로 윗면은 밤색이며 머리가 더 진하다. "
            "등에 검은 줄무늬가 있고 가슴과 배는 흰색이다. "
            "뺨은 흰색 바탕에 짙은 반점이 있다."
        ),
        "note": (
            "연중 관찰된다. 사용자가 제시한 번식 시기는 "
            "4월 중순~8월 중순이다. 25~35℃는 활동 한계로 "
            "해석할 수 없어 기온 조합 계산에서 제외했다."
        ),
    },
}


# =====================================================
# 3. 데이터 검사와 환경 계산
# =====================================================

def validate_database():
    for species, info in SPECIES_DB.items():
        hab = info["hab"]

        if len(hab) != len(HABITATS):
            raise ValueError(f"{species}: 서식지 점수가 4개가 아닙니다.")

        if not all(
            math.isfinite(float(v)) and 0 <= float(v) <= 10
            for v in hab
        ):
            raise ValueError(f"{species}: 서식지 점수는 0~10이어야 합니다.")

        temp = info["temp"]

        if temp is not None:
            if (
                len(temp) != 2
                or not all(math.isfinite(float(v)) for v in temp)
                or temp[0] > temp[1]
            ):
                raise ValueError(f"{species}: 기온 범위를 확인하세요.")


def calculate_environment(values, include_conditional=False):
    positive = {}

    for species, value in values.items():
        value = float(value)

        if (
            species not in SPECIES_DB
            or not math.isfinite(value)
            or not 0 <= value <= 100
        ):
            raise ValueError("등록된 종의 점수를 0~100으로 입력하세요.")

        if value > 0:
            positive[species] = value

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

    def active_times(species):
        info = SPECIES_DB[species]
        slots = set(info["times"])

        if include_conditional:
            slots.update(info.get("conditional_times", []))

        return slots

    time_scores = [
        sum(
            weight * 100
            for species, weight in weights
            if slot in active_times(species)
        )
        for slot in TIME_SLOTS
    ]

    # 중요: 0~10점이므로 10으로 나눈 후 100점 척도로 표시
    habitat_scores = [
        sum(
            weight
            * SPECIES_DB[species]["hab"][index]
            / 10.0
            * 100.0
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

    # 기온 자료가 있는 종만 사용하고 그 안에서 다시 가중평균
    temp_rows = [
        (species, weight)
        for species, weight in weights
        if SPECIES_DB[species]["temp"] is not None
    ]

    coverage = sum(weight for _, weight in temp_rows)

    low = None
    high = None

    if coverage > 0:
        low = sum(
            weight * SPECIES_DB[species]["temp"][0]
            for species, weight in temp_rows
        ) / coverage

        high = sum(
            weight * SPECIES_DB[species]["temp"][1]
            for species, weight in temp_rows
        ) / coverage

    if not candidates:
        judgment = "등록 자료로는 서식 환경 후보를 표시할 수 없습니다."
    elif len(candidates) == 1:
        judgment = (
            "현재 입력과 잠정 서식지 점수에서 상대적으로 지지되는 "
            "환경입니다. 실제 녹음 장소로 확정할 수는 없습니다."
        )
    else:
        judgment = (
            "여러 환경이 비슷하게 지지됩니다. "
            "현재 입력만으로 실제 장소를 구분하기 어렵습니다."
        )

    return {
        "raw": positive,
        "weights": weights,
        "months": month_scores,
        "times": time_scores,
        "low": low,
        "high": high,
        "temp_coverage": coverage,
        "habitat_scores": habitat_scores,
        "candidates": candidates,
        "judgment": judgment,
        "conditional": include_conditional,
        "shannon": -sum(
            weight * math.log(weight)
            for _, weight in weights
        ),
        "simpson": sum(
            weight ** 2
            for _, weight in weights
        ),
    }


# =====================================================
# 4. 입력 초기화·예시·텍스트 붙여넣기
# =====================================================

def reset_inputs():
    for species in SPECIES_DB:
        st.session_state[PREFIX + "score_" + species] = 0.0

    for group in GROUPS:
        st.session_state[PREFIX + "bg_" + group] = 0.0

    st.session_state[PREFIX + "paste"] = ""
    st.session_state.pop(PREFIX + "saved", None)
    st.session_state.pop(PREFIX + "message", None)


def load_example():
    reset_inputs()
    st.session_state[PREFIX + "score_털매미"] = 99.89
    st.session_state[PREFIX + "score_애매미"] = 0.02
    st.session_state[PREFIX + "bg_매미"] = 0.09


def apply_paste():
    text = st.session_state.get(PREFIX + "paste", "")
    parsed = {}

    # 새 CNN의 합쳐진 까마귀 항목도 수동 사전에 연결
    text = text.replace("큰부리까마귀+송장까마귀", "까마귀")

    for species in SPECIES_DB:
        pattern = (
            r"(?<![가-힣A-Za-z])"
            + re.escape(species)
            + r"""(?![가-힣A-Za-z])[\s*`"'：:=()%\-]*"""
            + r"([+-]?\d+(?:\.\d+)?)"
        )

        matches = re.findall(pattern, text)

        if len(matches) > 1:
            st.session_state[PREFIX + "message"] = (
                False,
                f"{species} 점수가 여러 번 나옵니다. 한 번만 입력하세요.",
            )
            return

        if matches:
            value = float(matches[0])

            if not math.isfinite(value) or not 0 <= value <= 100:
                st.session_state[PREFIX + "message"] = (
                    False,
                    f"{species}: 점수는 0~100이어야 합니다.",
                )
                return

            parsed[species] = value

    if not parsed:
        st.session_state[PREFIX + "message"] = (
            False,
            "예: 털매미 99.89%처럼 종 이름 뒤에 점수를 입력하세요.",
        )
        return

    for species in SPECIES_DB:
        st.session_state[PREFIX + "score_" + species] = float(
            parsed.get(species, 0.0)
        )

    for group in GROUPS:
        st.session_state[PREFIX + "bg_" + group] = 0.0

    st.session_state.pop(PREFIX + "saved", None)

    st.session_state[PREFIX + "message"] = (
        True,
        f"{len(parsed)}종의 점수를 반영했습니다.",
    )


# =====================================================
# 5. 기존 매미 MFCC 모델 연결
#
# 주의: 이 기능은 아래 세 파일로 구성된 기존 매미 모델 전용.
# 새로 만든 새소리 CNN 모델과는 다른 형식입니다.
# =====================================================

def find_model_files():
    folders = [
        BASE_DIR / "models" / "cicada",
        BASE_DIR,
    ]

    model_names = [
        "model.keras",
        "걸생 베타 매미 종 동정 모델.keras",
    ]

    for folder in folders:
        for name in model_names:
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

    categories_array = np.load(
        classes_path,
        allow_pickle=False,
    )

    with np.load(config_path, allow_pickle=False) as config:
        sample_rate = int(config["sample_rate"].item())
        n_mfcc = int(config["n_mfcc"].item())

    if categories_array.ndim != 1 or len(categories_array) < 2:
        raise ValueError("카테고리 목록이 올바르지 않습니다.")

    categories = [
        value.decode("utf-8") if isinstance(value, bytes)
        else str(value)
        for value in categories_array
    ]

    if len(categories) != len(set(categories)):
        raise ValueError("카테고리 이름이 중복됩니다.")

    if sample_rate <= 0 or n_mfcc <= 0:
        raise ValueError("음원 전처리 설정을 확인하세요.")

    if len(model.input_shape) != 2:
        raise ValueError("MFCC 평균값을 입력받는 매미 모델이 필요합니다.")

    if model.input_shape[-1] != n_mfcc:
        raise ValueError("모델 입력과 MFCC 개수가 다릅니다.")

    if model.output_shape[-1] != len(categories):
        raise ValueError("모델 출력과 카테고리 수가 다릅니다.")

    allowed = {
        name
        for name, info in SPECIES_DB.items()
        if info["group"] == "매미"
    } | {"배경소리"}

    if not set(categories).issubset(allowed):
        raise ValueError("기존 매미 분류 모델을 연결해 주세요.")

    return model, categories, sample_rate, n_mfcc


def classify_audio(uploaded, paths):
    import librosa

    suffix = Path(uploaded.name).suffix.lower()

    if suffix not in {".mp3", ".wav"}:
        raise ValueError("MP3 또는 WAV 파일을 사용하세요.")

    if uploaded.size > 20 * 1024 * 1024:
        raise ValueError("20MB 이하 파일을 사용하세요.")

    file_version = tuple(
        (path.stat().st_mtime_ns, path.stat().st_size)
        for path in paths
    )

    model, categories, sample_rate, n_mfcc = load_model_files(
        str(paths[0]),
        str(paths[1]),
        str(paths[2]),
        file_version,
    )

    audio, sr = librosa.load(
        io.BytesIO(uploaded.getvalue()),
        sr=sample_rate,
        mono=True,
        duration=61,
    )

    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise ValueError("음원이 비어 있거나 손상됐습니다.")

    duration = len(audio) / sr

    if duration > 60:
        raise ValueError("60초 이하 음원을 사용하세요.")

    if np.max(np.abs(audio)) <= 1e-8:
        raise ValueError("무음 파일은 분석할 수 없습니다.")

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
        raise ValueError("Softmax 모델 출력인지 확인하세요.")

    return {
        category: float(score * 100)
        for category, score in zip(categories, scores)
    }, duration


# =====================================================
# 6. 결과 표시
# =====================================================

def show_result(result):
    st.divider()
    st.subheader("생태환경 참고 결과")

    st.markdown("### 서식 환경 후보")
    st.write(
        " · ".join(result["candidates"])
        if result["candidates"]
        else "판단 불가"
    )
    st.info(result["judgment"])

    habitat_table = pd.DataFrame({
        "서식 환경": HABITATS,
        "참고 점수 (0~100)": [
            round(value, 1)
            for value in result["habitat_scores"]
        ],
        "후보": [
            "후보" if name in result["candidates"] else "—"
            for name in HABITATS
        ],
    }).sort_values(
        "참고 점수 (0~100)",
        ascending=False,
    )

    st.dataframe(
        habitat_table,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "서식지 점수는 확률이나 거주 비율이 아닙니다. "
        "환경별 점수의 합계를 100으로 맞추지 않습니다."
    )

    with st.expander("서식 환경 계산 방법"):
        st.code(
            "반영 비중 = 해당 종 입력 점수 / 전체 생물 종 입력 점수 합\n"
            "환경 점수 = Σ(반영 비중 × 종별 서식지 점수 / 10 × 100)",
            language="text",
        )

        st.write(
            f"최고점과 {CANDIDATE_MARGIN:.0f}점 이내인 환경을 "
            "함께 후보로 표시합니다."
        )

        st.caption(
            "예: 말매미만 입력하면 "
            "도심 100 / 산림 60 / 농경지 40 / 물가 20입니다. "
            "도심에 있을 확률이 100%라는 뜻은 아닙니다."
        )

        st.warning(
            "서식지 점수와 후보 기준은 검증된 예측 모델이 아닙니다. "
            "특히 초지/농경지와 습지/물가는 식생 조건에 따라 "
            "같은 분류 안에서도 큰 차이가 생깁니다."
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
        st.caption("등록 시기와의 일치")
        st.write(" · ".join(best_months))

    with columns[1]:
        st.caption("등록 활동 시간과의 일치")
        st.write(" · ".join(best_times))

    with columns[2]:
        st.caption("등록 기온 범위의 가중평균")

        if result["low"] is None:
            st.write("계산 가능한 자료 없음")
        else:
            st.write(
                f'{result["low"]:.1f} ~ '
                f'{result["high"]:.1f}℃'
            )

    if result["temp_coverage"] < 1 - 1e-9:
        st.caption(
            "기온 범위가 없는 종을 제외했습니다. "
            f"기온 계산에 포함된 반영 비중: "
            f"{result['temp_coverage'] * 100:.1f}%"
        )

    if result["conditional"]:
        st.caption(
            "조건부 낮 활동을 포함했습니다. "
            "해당 조건이 모든 종에 실제로 충족됐다는 뜻은 아닙니다."
        )

    dominant, weight = result["weights"][0]

    st.success(
        f"가장 크게 반영된 종: {dominant} "
        f"· 반영 비중 {weight * 100:.1f}%"
    )

    st.warning(
        "시기·시간·기온은 입력된 생태 정보를 조합한 참고값입니다. "
        "녹음 날짜·시간·온도를 실제로 측정하거나 추정한 결과가 아닙니다. "
        "서로 다른 계절의 종이 섞여도 계산상 결과가 나올 수 있습니다."
    )

    with st.expander("종별 비중과 상세 점수"):
        st.dataframe(
            pd.DataFrame([
                {
                    "종": species,
                    "그룹": SPECIES_DB[species]["group"],
                    "입력 점수": round(result["raw"][species], 2),
                    "반영 비중 (%)": round(weight * 100, 2),
                }
                for species, weight in result["weights"]
            ]),
            hide_index=True,
            use_container_width=True,
        )

        st.write("월별 등록 정보 일치 점수")

        st.dataframe(
            pd.DataFrame({
                "월": [f"{month}월" for month in range(1, 13)],
                "점수 (0~100)": result["months"],
            }),
            hide_index=True,
            use_container_width=True,
        )

        st.write("시간대별 등록 정보 일치 점수")

        st.dataframe(
            pd.DataFrame({
                "시간대": TIME_SLOTS,
                "점수 (0~100)": result["times"],
            }),
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            f"점수 분포 H′: {result['shannon']:.4f} / "
            f"점수 집중도 D: {result['simpson']:.4f}"
        )

        st.info(
            "이 값들은 분류 점수 분포를 요약한 것입니다. "
            "개체수 기반 생물다양성 지수가 아닙니다. "
            "서로 다른 분류 모델의 점수도 직접 비교 가능하다고 "
            "보장되지 않습니다."
        )

    with st.expander("반영된 종의 생태 정보"):
        for species, _ in result["weights"]:
            show_species(species)


def show_species(species):
    info = SPECIES_DB[species]

    st.markdown(f"**{species} · {info['group']}**")

    st.write(
        "1. 등록 출현 시기: "
        + ", ".join(f"{month}월" for month in info["months"])
    )

    st.write(
        "2. 주 활동 시간: "
        + " · ".join(info["times"])
    )

    if info.get("condition"):
        st.caption("조건부 활동: " + info["condition"])

    if info["temp"] is None:
        st.write("3. 참고 기온 범위: 수치 계산에서 제외")
    else:
        st.write(
            f"3. 등록 참고 기온: "
            f"{info['temp'][0]} ~ {info['temp'][1]}℃"
        )

    st.write("4. 잠정 서식지 적합도 (0~10)")

    st.dataframe(
        pd.DataFrame({
            "환경": HABITATS,
            "점수": info["hab"],
        }),
        hide_index=True,
        use_container_width=True,
    )

    st.write("5. 울음소리: " + info["sound"])
    st.write("6. 외형·생태: " + info["desc"])

    if info.get("note"):
        st.warning(info["note"])

    st.divider()


# =====================================================
# 7. 수동 입력 화면
# =====================================================

def manual_page():
    for species in SPECIES_DB:
        st.session_state.setdefault(
            PREFIX + "score_" + species,
            0.0,
        )

    for group in GROUPS:
        st.session_state.setdefault(
            PREFIX + "bg_" + group,
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

    with st.expander("분류 결과 텍스트 붙여넣기"):
        st.text_area(
            "종 이름과 점수",
            placeholder=(
                "털매미 99.89%\n"
                "애매미 0.02%\n"
                "배경소리 0.09%"
            ),
            key=PREFIX + "paste",
        )

        st.button(
            "점수 반영",
            on_click=apply_paste,
        )

        st.caption(
            "생물 종 점수만 가져옵니다. "
            "배경소리는 아래 그룹별 칸에 직접 입력하세요. "
            "소수점은 99.89처럼 점(.)으로 입력합니다."
        )

        message = st.session_state.get(PREFIX + "message")

        if message:
            success, text = message
            if success:
                st.success(text)
            else:
                st.error(text)

    with st.form(PREFIX + "form"):
        columns = st.columns(3)

        for column, group in zip(columns, GROUPS):
            with column:
                st.markdown(f"### {group}")

                for species, info in SPECIES_DB.items():
                    if info["group"] == group:
                        st.number_input(
                            f"{species} (%)",
                            min_value=0.0,
                            max_value=100.0,
                            step=0.01,
                            format="%.2f",
                            key=PREFIX + "score_" + species,
                        )

                st.number_input(
                    f"{group} 배경소리 (%)",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.01,
                    format="%.2f",
                    key=PREFIX + "bg_" + group,
                )

        conditional = st.checkbox(
            "비·흐림·번식기 등 조건부 낮 활동도 포함",
            value=False,
        )

        submitted = st.form_submit_button(
            "생태환경 참고 결과 계산",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        values = {
            species: float(
                st.session_state[PREFIX + "score_" + species]
            )
            for species in SPECIES_DB
        }

        backgrounds = {
            group: float(
                st.session_state[PREFIX + "bg_" + group]
            )
            for group in GROUPS
        }

        try:
            calculate_environment(values, conditional)

            st.session_state[PREFIX + "saved"] = {
                "values": values,
                "backgrounds": backgrounds,
                "conditional": conditional,
            }
        except ValueError as exc:
            st.session_state.pop(PREFIX + "saved", None)
            st.error(str(exc))

    saved = st.session_state.get(PREFIX + "saved")

    if saved:
        active_groups = {
            SPECIES_DB[species]["group"]
            for species, value in saved["values"].items()
            if value > 0
        }

        for group in active_groups:
            species_total = sum(
                value
                for species, value in saved["values"].items()
                if SPECIES_DB[species]["group"] == group
            )

            if saved["backgrounds"][group] >= species_total:
                st.warning(
                    f"{group}: 배경소리 점수가 생물 종 점수 합계 이상입니다. "
                    "작은 오분류 점수가 확대 반영될 수 있습니다."
                )

        if len(active_groups) > 1:
            st.warning(
                "서로 다른 그룹 모델의 점수를 합쳤습니다. "
                "모델별 점수 보정이 없으므로 상대적 비중이 왜곡될 수 있습니다."
            )

        st.caption(
            "배경소리는 환경 계산에서 제외합니다. "
            "남은 생물 점수를 다시 정규화하므로 "
            "반영 비중이 실제 검출 신뢰도를 뜻하지는 않습니다."
        )

        show_result(
            calculate_environment(
                saved["values"],
                saved["conditional"],
            )
        )


# =====================================================
# 8. 매미 음원 분석 화면
# =====================================================

def audio_page():
    st.subheader("기존 매미 모델로 음원 분석")

    st.caption(
        "MP3·WAV / 최대 20MB / 60초 이하. "
        "파일 전체의 대표 소리를 분류합니다."
    )

    st.info(
        "이 탭은 기존 MFCC 평균 기반 매미 모델 전용입니다. "
        "새소리 CNN 모델은 연결하지 않습니다. "
        "새소리 CNN 결과는 점수 직접 입력 탭에 붙여 넣으세요."
    )

    paths = find_model_files()

    if paths is None:
        st.warning(
            "매미 모델 파일이 없습니다. "
            "점수 직접 입력 기능은 그대로 사용할 수 있습니다."
        )

        with st.expander("모델 파일 위치"):
            st.code(
                "models/\n"
                "└── cicada/\n"
                "    ├── model.keras\n"
                "    ├── classes.npy\n"
                "    └── feature_config.npz",
                language="text",
            )

            st.caption(
                "기존 모델명 '걸생 베타 매미 종 동정 모델.keras'도 "
                "지원합니다. 세 파일은 같은 학습 결과여야 합니다."
            )

            st.warning(
                "이 연결 코드는 MFCC 평균을 바로 입력하는 모델용입니다. "
                "외부 StandardScaler가 필요한 모델은 "
                "그 Scaler까지 연결하지 않으면 올바르게 예측할 수 없습니다."
            )

    uploaded = st.file_uploader(
        "분석할 매미 녹음",
        type=["mp3", "wav"],
        key=PREFIX + "audio",
    )

    conditional = st.checkbox(
        "조건부 낮 활동 포함",
        key=PREFIX + "audio_conditional",
    )

    if uploaded is None:
        return

    if uploaded.size > 20 * 1024 * 1024:
        st.error("20MB 이하 음원을 사용하세요.")
        return

    st.audio(uploaded)

    if st.button(
        "음원 분석하기",
        type="primary",
        disabled=paths is None,
    ):
        try:
            with st.spinner("소리를 분석하고 있습니다..."):
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
                    columns=["카테고리", "모델 점수 (%)"],
                ),
                hide_index=True,
                use_container_width=True,
            )

            st.warning(
                "모델 점수는 실제 정답 확률이 아닙니다. "
                "학습하지 않은 소리도 높은 점수로 오분류할 수 있습니다."
            )

            best_species = max(scores, key=scores.get)

            if best_species == "배경소리":
                st.info(
                    "배경소리가 최고 점수여서 환경 계산을 보류했습니다."
                )
                return

            values = {
                species: score
                for species, score in scores.items()
                if species in SPECIES_DB
            }

            st.caption(
                "배경소리를 제외한 생물 점수를 다시 정규화합니다."
            )

            show_result(
                calculate_environment(
                    values,
                    conditional,
                )
            )

        except Exception as exc:
            st.error(f"분석하지 못했습니다: {exc}")


# =====================================================
# 9. 메인
# =====================================================

def main():
    validate_database()

    st.title("🌿 소리로 살펴보는 생태환경")
    st.success("V4 · 서식지 0~10점 반영")

    st.caption(
        "종별 음향 분류 점수와 등록된 생태 정보를 결합하는 "
        "규칙 기반 참고 도구입니다. "
        "현장 자료로 학습된 서식지 예측 모델은 아닙니다."
    )

    manual_tab, audio_tab, dictionary_tab = st.tabs([
        "점수 직접 입력",
        "매미 음원 분석",
        "전체 생태 사전",
    ])

    with manual_tab:
        manual_page()

    with audio_tab:
        audio_page()

    with dictionary_tab:
        st.warning(
            "서식지 점수는 사용자 제공·문헌 해석 기반 잠정값입니다. "
            "공식 서식 비율이나 실측 밀도가 아닙니다. "
            "시기·기온·외형 설명 역시 별도 검증이 필요합니다."
        )

        rows = [
            {
                "종": species,
                "그룹": info["group"],
                **dict(zip(HABITATS, info["hab"])),
            }
            for species, info in SPECIES_DB.items()
        ]

        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
        )

        species = st.selectbox(
            "상세 정보를 볼 생물",
            list(SPECIES_DB),
        )

        show_species(species)

    st.divider()

    st.caption(
        "음향 특징과 외형 설명은 사전 표시용이며 "
        "서식지 계산식에는 직접 사용하지 않습니다."
    )

    st.caption(
        "온라인 배포 시 음원은 서버에서 처리됩니다. "
        "이 앱 코드는 업로드 음원을 영구 저장하지 않습니다."
    )


if __name__ == "__main__":
    main()
