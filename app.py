from pathlib import Path
import math
import re
import tempfile

import numpy as np
import pandas as pd
import streamlit as st


# ====================================================
# 1. 페이지 설정
# ====================================================

st.set_page_config(
    page_title="Habitat Prediction Model",
    page_icon="🌿",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent

GROUPS = ["개구리", "매미", "새"]
TIME_SLOTS = ["새벽", "오전", "한낮", "저녁", "밤"]

HABITATS = [
    "도심/공원",
    "산림/숲",
    "초지/농경지",
    "습지/물가",
]


# ====================================================
# 2. 생태 정보
#
# 기존에 제공한 코드의 수치를 사용합니다.
# 이 자료는 실제 관측값이 아니며 별도 검증이 필요합니다.
#
# 순서:
# 종, 그룹, 출현 월, 활동 시간, 기온 범위,
# 서식 환경 코드, 울음소리, 설명
# ====================================================

SPECIES_ROWS = [
    (
        "두꺼비", "개구리",
        [2, 3, 4],
        ["오전", "저녁", "밤"],
        [1.5, 12.3],
        [1, 1, 1, 1],
        "콕— 콕— 콕—",
        "갈색 몸에 돌기가 많고 눈 뒤에 독샘이 있다. "
        "평소 육상에서 생활하다 봄철 물가에서 번식한다.",
    ),
    (
        "청개구리", "개구리",
        [4, 5, 6],
        ["오전", "저녁", "밤"],
        [12.3, 21.6],
        [1, 1, 1, 1],
        "꽥 꽥 꽥 / 깩 깩 깩",
        "몸길이 약 2.5~4cm. 발가락 끝에 흡반이 있어 "
        "풀과 나무를 잘 타고 논과 웅덩이에서 번식한다.",
    ),
    (
        "참개구리", "개구리",
        [4, 5, 6],
        ["저녁", "밤"],
        [12.3, 21.6],
        [1, 0, 1, 1],
        "까르르르르 / 꾸르륵 꾸르륵",
        "몸길이 약 6~9cm. 등에 검은 반점과 밝은 중앙 줄이 있고 "
        "논과 연못 등에서 생활한다.",
    ),
    (
        "황소개구리", "개구리",
        [4, 5, 6, 7],
        ["저녁", "밤"],
        [12.3, 24.8],
        [1, 0, 1, 1],
        "우음— 우음— / 우우웅",
        "몸길이 약 15~20cm. 눈 뒤 고막이 뚜렷하며 "
        "저수지와 연못 등에서 생활한다.",
    ),
    (
        "북방산개구리", "개구리",
        [2, 3, 4],
        ["한낮", "저녁", "밤"],
        [1.5, 12.3],
        [1, 1, 1, 1],
        "호르르릉 / 오그그그그",
        "몸길이 약 5~8.5cm. 산림에서 생활하고 이른 봄 "
        "논이나 계곡 주변 물웅덩이에서 번식한다.",
    ),
    (
        "맹꽁이", "개구리",
        [5, 6, 7, 8],
        ["저녁", "밤"],
        [17.5, 25.3],
        [1, 1, 1, 1],
        "맹— 맹— / 꽁— 꽁—",
        "몸길이 약 3.5~5.5cm. 몸통이 둥글고 팔다리가 짧다. "
        "장마철 물웅덩이에서 번식하며 비가 오면 낮에도 운다.",
    ),
    (
        "유지매미", "매미",
        [6, 7, 8, 9],
        ["오전", "한낮", "저녁"],
        [20.4, 28.2],
        [1, 1, 1, 0],
        "찍 끄르르르 / 딕! 쯔그르르르",
        "몸길이 34~36mm, 날개 끝까지 약 50~60mm.",
    ),
    (
        "애매미", "매미",
        [6, 7, 8, 9],
        ["오전", "한낮", "저녁"],
        [20.4, 28.2],
        [1, 1, 1, 0],
        "스삐오스 스삐오스 치르르르",
        "작은 크기를 가지며 해발 600m 이하에서 주로 서식한다.",
    ),
    (
        "참매미", "매미",
        [6, 7, 8, 9],
        ["새벽", "오전", "한낮", "저녁"],
        [20.4, 28.2],
        [1, 1, 1, 0],
        "맴맴맴",
        "머리와 가슴이 단단하고 배는 짧고 두툼하다.",
    ),
    (
        "쓰름매미", "매미",
        [7, 8],
        ["오전", "저녁"],
        [21.4, 29.9],
        [1, 1, 1, 0],
        "쓰름쓰름",
        "울음소리가 쓰름, 쓰름으로 들려 이름이 붙었다.",
    ),
    (
        "털매미", "매미",
        [6, 7, 8, 9],
        ["오전", "한낮", "저녁"],
        [15.0, 34.0],
        [1, 1, 1, 0],
        "찌이이이— / 쓰으으으—",
        "몸길이 약 20~24mm, 날개 끝까지 약 35~40mm. "
        "길게 울다가 음높이가 바뀐다.",
    ),
    (
        "말매미", "매미",
        [6, 7, 8, 9, 10],
        ["오전", "한낮", "밤"],
        [24.0, 30.0],
        [1, 1, 1, 0],
        "찌르르르",
        "우리나라 매미 중 가장 큰 종으로 한반도 전역에 서식한다.",
    ),
    (
        "물까치", "새",
        list(range(1, 13)),
        ["오전", "한낮", "저녁"],
        [-4.6, 24.9],
        [1, 1, 1, 0],
        "꺄악-꺅꺅꺅꺅꺅",
        "일반적으로 무리 지어 생활하는 텃새로 한반도 전역에 서식한다.",
    ),
    (
        "까마귀", "새",
        list(range(1, 13)),
        ["오전", "한낮", "저녁"],
        [-0.9, 25.1],
        [1, 1, 1, 1],
        "까악— 까악—",
        "몸길이 약 50cm. 검은 몸에 녹색 광택이 있고 "
        "산림·농경지·공원·하천 등에서 연중 생활한다.",
    ),
    (
        "멧비둘기", "새",
        list(range(1, 13)),
        ["오전", "한낮"],
        [-15.0, 35.0],
        [1, 1, 0, 0],
        "구, 구, 쿠-, 쿠-",
        "몸길이 약 33cm. 목 옆에 회색과 검은색 줄무늬 반점이 있다.",
    ),
    (
        "뻐꾸기", "새",
        [4, 5, 6, 7, 8, 9],
        ["오전", "한낮", "저녁", "밤"],
        [12.3, 25.3],
        [1, 1, 1, 1],
        "뻐꾹- / 삐삐삐삐",
        "몸길이 32~36cm. 눈과 눈테가 노란색이고 몸은 옅은 검은색이다.",
    ),
    (
        "까치", "새",
        list(range(1, 13)),
        ["오전", "한낮"],
        [-0.6, 25.3],
        [1, 1, 1, 0],
        "꺅-꺅-꺅-꺅",
        "전체 길이 약 46cm. 윗면은 푸른 광택의 검은색이고 배는 흰색이다.",
    ),
    (
        "참새", "새",
        list(range(1, 13)),
        ["오전", "한낮", "저녁"],
        [25.0, 35.0],
        [1, 0, 1, 0],
        "짹 짹",
        "몸길이 약 14.5cm. 윗면은 밤색이고 등에 검은 줄무늬가 있으며 "
        "가슴과 배는 흰색이다.",
    ),
]

SPECIES_DB = {}

for row in SPECIES_ROWS:
    SPECIES_DB[row[0]] = {
        "group": row[1],
        "months": row[2],
        "times": row[3],
        "temp": row[4],
        "hab": row[5],
        "sound": row[6],
        "desc": row[7],
    }


# ====================================================
# 3. 생태환경 계산
# ====================================================

def estimate_environment(values):
    for species, value in values.items():
        if (
            species not in SPECIES_DB
            or not math.isfinite(value)
            or not 0 <= value <= 100
        ):
            raise ValueError("점수는 0~100 사이의 숫자여야 합니다.")

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

    months = [
        sum(
            weight * 100
            for species, weight in weights
            if month in SPECIES_DB[species]["months"]
        )
        for month in range(1, 13)
    ]

    times = [
        sum(
            weight * 100
            for species, weight in weights
            if slot in SPECIES_DB[species]["times"]
        )
        for slot in TIME_SLOTS
    ]

    low = sum(
        weight * SPECIES_DB[species]["temp"][0]
        for species, weight in weights
    )

    high = sum(
        weight * SPECIES_DB[species]["temp"][1]
        for species, weight in weights
    )

    habitat = [
        sum(
            weight * SPECIES_DB[species]["hab"][index] * 100
            for species, weight in weights
        )
        for index in range(4)
    ]

    habitat_total = sum(habitat)

    ratios = [
        value / habitat_total * 100 if habitat_total else 0
        for value in habitat
    ]

    return {
        "raw": positive,
        "weights": weights,
        "months": months,
        "times": times,
        "low": low,
        "high": high,
        "habitat": habitat,
        "habitat_ratio": ratios,
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
# 4. 점수 붙여넣기 / 예시 / 초기화
# 콜백으로 입력 위젯이 만들어지기 전에 값을 바꿉니다.
# ====================================================

def clear_manual():
    for species in SPECIES_DB:
        st.session_state["score_" + species] = 0.0

    for group in GROUPS:
        st.session_state["background_" + group] = 0.0

    st.session_state["paste_text"] = ""
    st.session_state.pop("manual_result", None)
    st.session_state.pop("paste_message", None)


def load_example():
    clear_manual()

    st.session_state["score_털매미"] = 99.89
    st.session_state["score_애매미"] = 0.02
    st.session_state["background_매미"] = 0.09


def apply_paste():
    text = st.session_state.get("paste_text", "")
    parsed = {}

    for species in SPECIES_DB:
        pattern = (
            re.escape(species)
            + r"\s+([+-]?\d+(?:\.\d+)?)"
        )

        match = re.search(pattern, text)

        if match:
            value = float(match.group(1))

            if not 0 <= value <= 100:
                st.session_state["paste_message"] = (
                    False,
                    f"{species}: 점수는 0~100 사이여야 합니다.",
                )
                return

            parsed[species] = value

    if not parsed:
        st.session_state["paste_message"] = (
            False,
            "종 이름 뒤에 점수를 넣어 주세요. 예: 털매미 99.89%",
        )
        return

    # 이전 입력값과 섞이지 않도록 기존 값을 비웁니다.
    for species in SPECIES_DB:
        st.session_state["score_" + species] = float(
            parsed.get(species, 0)
        )

    for group in GROUPS:
        st.session_state["background_" + group] = 0.0

    st.session_state.pop("manual_result", None)

    st.session_state["paste_message"] = (
        True,
        f"{len(parsed)}종의 점수를 반영했습니다.",
    )


# ====================================================
# 5. 매미 모델 파일 찾기
# ====================================================

def find_model_files():
    folders = [
        BASE_DIR / "models" / "cicada",
        BASE_DIR,
    ]

    filenames = [
        "model.keras",
        "걸생 베타 매미 종 동정 모델.keras",
    ]

    for folder in folders:
        for filename in filenames:
            paths = (
                folder / filename,
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
    # TensorFlow는 음원 분석을 사용할 때만 불러옵니다.
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

    suffix = Path(uploaded.name).suffix.lower()

    with tempfile.TemporaryDirectory() as folder:
        temporary_path = Path(folder) / ("audio" + suffix)
        temporary_path.write_bytes(uploaded.getvalue())

        audio, sr = librosa.load(
            str(temporary_path),
            sr=sample_rate,
            mono=True,
            duration=61,
        )

    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise ValueError("음원이 비어 있거나 잘못된 값이 있습니다.")

    duration = len(audio) / sr

    if duration > 60:
        raise ValueError("60초 이하의 음원을 올려 주세요.")

    if np.max(np.abs(audio)) == 0:
        raise ValueError("완전히 무음인 음원은 분석할 수 없습니다.")

    # 기존 학습 코드와 같은 특징 추출 방식
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
    )

    feature = np.mean(mfcc, axis=1).astype(np.float32)

    if not np.all(np.isfinite(feature)):
        raise ValueError("음원 특징 추출에 실패했습니다.")

    # 정규화는 저장된 모델 내부에서 처리합니다.
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
        raise ValueError("모델 예측 점수가 올바르지 않습니다.")

    return {
        category: float(score * 100)
        for category, score in zip(categories, scores)
    }, duration


# ====================================================
# 7. 결과 표시
# ====================================================

def show_result(result):
    st.divider()
    st.subheader("생태환경 참고 결과")

    best_months = [
        f"{index + 1}월"
        for index, value in enumerate(result["months"])
        if abs(value - max(result["months"])) < 0.01
    ]

    best_times = [
        slot
        for slot, value in zip(TIME_SLOTS, result["times"])
        if abs(value - max(result["times"])) < 0.01
    ]

    columns = st.columns(3)

    with columns[0]:
        st.caption("시기")
        st.write(" · ".join(best_months))

    with columns[1]:
        st.caption("활동 시간")
        st.write(" · ".join(best_times))

    with columns[2]:
        st.caption("참고 기온 범위")
        st.write(
            f'{result["low"]:.1f} ~ {result["high"]:.1f} °C'
        )

    species, weight = result["weights"][0]

    st.success(
        f"가장 크게 반영된 종: {species}"
        f" · 반영 비중 {weight * 100:.1f}%"
    )

    st.caption(
        "기존 생태 정보를 조합한 참고값입니다. "
        "실제 녹음 시기·기온·서식지의 관측값이나 확률이 아닙니다."
    )

    with st.expander("종별 비중과 상세 계산"):
        species_table = pd.DataFrame([
            {
                "종": species,
                "그룹": SPECIES_DB[species]["group"],
                "입력 점수 (%)": round(result["raw"][species], 2),
                "반영 비중 (%)": round(weight * 100, 2),
            }
            for species, weight in result["weights"]
        ])

        st.dataframe(
            species_table,
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("**월별 참고 적합도**")

        st.dataframe(
            pd.DataFrame({
                "월": [
                    f"{month}월"
                    for month in range(1, 13)
                ],
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

        st.markdown("**서식 환경 비교**")

        st.dataframe(
            pd.DataFrame({
                "서식 환경": HABITATS,
                "적합도 (%)": result["habitat"],
                "상대 비중 (%)": result["habitat_ratio"],
            }),
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            f'점수 기반 H′: {result["shannon"]:.4f}'
            f' / 점수 기반 D: {result["simpson"]:.4f}'
        )

        st.info(
            "H′와 D는 입력 점수의 분포를 계산한 값입니다. "
            "실제 개체수 기반 생물다양성·우점도 지수가 아닙니다. "
            "서로 다른 모델의 점수도 같은 기준으로 "
            "비교 가능하다고 보장할 수 없습니다."
        )

    with st.expander("입력된 종의 생태 사전"):
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
                "서식 환경: "
                + ", ".join(habitat_names)
            )

            st.write("울음소리: " + info["sound"])
            st.write(info["desc"])
            st.divider()


# ====================================================
# 8. 점수 직접 입력 화면
# ====================================================

def manual_page():
    for species in SPECIES_DB:
        st.session_state.setdefault(
            "score_" + species,
            0.0,
        )

    for group in GROUPS:
        st.session_state.setdefault(
            "background_" + group,
            0.0,
        )

    first, second = st.columns(2)

    with first:
        st.button(
            "예시 넣기",
            on_click=load_example,
            use_container_width=True,
        )

    with second:
        st.button(
            "초기화",
            on_click=clear_manual,
            use_container_width=True,
        )

    with st.expander("모델 결과 텍스트 붙여넣기"):
        st.text_area(
            "모델에서 복사한 결과",
            placeholder=(
                "1 털매미 99.89%\n"
                "2 배경소리 0.09%\n"
                "3 애매미 0.02%"
            ),
            key="paste_text",
        )

        st.button(
            "점수 반영",
            on_click=apply_paste,
        )

        st.caption(
            "생물 종 점수만 반영합니다. "
            "배경소리 점수는 아래에서 직접 입력하세요."
        )

        if "paste_message" in st.session_state:
            success, message = st.session_state["paste_message"]

            if success:
                st.success(message)
            else:
                st.error(message)

    with st.form("species_form"):
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
                        key="score_" + species,
                    )

                st.number_input(
                    f"{group} 배경소리 (%)",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.01,
                    format="%.2f",
                    key="background_" + group,
                )

        submitted = st.form_submit_button(
            "생태환경 추정하기",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            values = {
                species: float(
                    st.session_state["score_" + species]
                )
                for species in SPECIES_DB
            }

            st.session_state["manual_result"] = (
                estimate_environment(values)
            )

        except ValueError as error:
            st.session_state.pop("manual_result", None)
            st.error(str(error))

    if "manual_result" in st.session_state:
        if any(
            st.session_state["background_" + group] > 0
            for group in GROUPS
        ):
            st.caption(
                "배경소리 점수는 계산에서 제외됩니다. "
                "배경 점수가 높다면 실제 종 검출 여부를 "
                "먼저 확인해 주세요."
            )

        show_result(st.session_state["manual_result"])


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

        with st.expander("모델 연결 방법"):
            st.code(
                "models/\n"
                "└── cicada/\n"
                "    ├── model.keras\n"
                "    ├── classes.npy\n"
                "    └── feature_config.npz",
                language="text",
            )

            st.write(
                "기존 모델명 "
                "'걸생 베타 매미 종 동정 모델.keras'도 지원합니다."
            )

            st.caption(
                "세 파일은 같은 학습에서 나온 파일이어야 합니다. "
                "개구리·새 자동 분류는 아직 연결하지 않았습니다."
            )

    uploaded = st.file_uploader(
        "녹음 파일 선택",
        type=["wav", "mp3"],
        key="uploaded_audio",
    )

    if uploaded is None:
        return

    if uploaded.size > 20 * 1024 * 1024:
        st.error("20MB 이하의 파일을 올려 주세요.")
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

            score_table = pd.DataFrame(
                sorted(
                    scores.items(),
                    key=lambda item: item[1],
                    reverse=True,
                ),
                columns=["종", "모델 점수 (%)"],
            )

            st.dataframe(
                score_table,
                hide_index=True,
                use_container_width=True,
            )

            st.warning(
                "예측 점수는 정답 확률이나 실제 정확도가 아닙니다. "
                "학습하지 않은 소리도 등록된 종류로 분류될 수 있습니다. "
                "여러 종의 존재를 개별적으로 검출하는 모델은 아닙니다."
            )

            best_species = max(scores, key=scores.get)

            if best_species == "배경소리":
                st.info(
                    "배경소리 점수가 가장 높아 환경 추정을 보류했습니다. "
                    "생물 소리가 뚜렷한 녹음을 사용해 주세요."
                )
                return

            biological_scores = {
                species: score
                for species, score in scores.items()
                if species in SPECIES_DB
            }

            st.caption(
                "배경소리를 제외하고 나머지 점수를 "
                "다시 100%로 환산해 계산합니다."
            )

            show_result(
                estimate_environment(biological_scores)
            )

        except Exception as error:
            st.error(f"분석하지 못했습니다: {error}")


# ====================================================
# 10. 메인 화면
# ====================================================

def main():
    st.title("소리로 살펴보는 생태환경")

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
        "생태 자료는 사용자가 제공한 원본 수치이며 별도 검증이 필요합니다."
    )

    st.caption(
        "온라인 배포 시 음원은 서버로 전송되어 처리됩니다. "
        "이 앱 코드는 업로드 음원을 영구 저장하지 않습니다."
    )


if __name__ == "__main__":
    main()
