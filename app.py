# app.py (Azure + GPT版 - Streamlit Cloud対応)
import streamlit as st
import pandas as pd
import os
import csv
import json
import uuid
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from pydub import AudioSegment
import gdown

# Azure SDKとOpenAIのインポートを条件付きに
try:
    import azure.cognitiveservices.speech as speechsdk
    from openai import OpenAI
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    st.error("必要なパッケージがインストールされていません。requirements.txtを確認してください。")

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_api_keys():
    """APIキーを環境変数またはStreamlit Secretsから取得"""
    # 環境変数を優先（ローカル実行時）
    azure_key = os.getenv("AZURE_SPEECH_KEY", "")
    azure_region = os.getenv("AZURE_SPEECH_REGION", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    
    # Streamlit Secretsから取得（デプロイ時）
    if not azure_key:
        try:
            azure_key = st.secrets.get("AZURE_SPEECH_KEY", "")
            azure_region = st.secrets.get("AZURE_SPEECH_REGION", "")
            openai_key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception as e:
            st.error(f"APIキーの取得に失敗しました: {e}")
            st.info("Streamlit Cloudの場合は、Settings > Secrets でAPIキーを設定してください。")
    
    return azure_key, azure_region, openai_key

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def convert_to_wav(input_path: Path, output_path: Path):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    audio.export(output_path, format="wav")
    return output_path

def download_audio_from_youtube(url: str, out_dir: Path) -> Path:
    ensure_dir(out_dir)
    out_path = out_dir / f"{uuid.uuid4().hex}.mp3"
    cmd = ["yt-dlp", "-x", "--audio-format", "mp3", "--audio-quality", "128k", "-o", str(out_path), url]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    wav_path = out_dir / f"{uuid.uuid4().hex}.wav"
    return convert_to_wav(out_path, wav_path)

def download_from_google_drive(url: str, out_dir: Path) -> Path:
    ensure_dir(out_dir)
    st.warning("Google Driveは現在MP3ファイルのみ対応")
    out_path = out_dir / f"{uuid.uuid4().hex}.mp3"
    gdown.download(url, str(out_path), quiet=False, fuzzy=True)
    wav_path = out_dir / f"{uuid.uuid4().hex}.wav"
    return convert_to_wav(out_path, wav_path)

def extract_audio_from_file(uploaded_file, out_dir: Path) -> Path:
    ensure_dir(out_dir)
    file_ext = uploaded_file.name.split('.')[-1].lower()
    if file_ext != 'mp3':
        raise ValueError("現在MP3ファイルのみ対応しています。")
    out_path = out_dir / f"{uuid.uuid4().hex}.mp3"
    temp_path = out_dir / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    import shutil
    shutil.copy(temp_path, out_path)
    temp_path.unlink(missing_ok=True)
    wav_path = out_dir / f"{uuid.uuid4().hex}.wav"
    return convert_to_wav(out_path, wav_path)

def azure_speech_to_text(audio_path: Path, region: str, key: str) -> str:
    if not AZURE_AVAILABLE:
        raise ValueError("Azure SDKが利用できません")
    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    audio_config = speechsdk.audio.AudioConfig(filename=str(audio_path))
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, language="en-US", audio_config=audio_config)
    result = recognizer.recognize_once()
    if result.reason == speechsdk.ResultReason.NoMatch:
        raise ValueError("No speech could be recognized.")
    return result.text

def azure_pronunciation_assess(audio_path: Path, region: str, key: str, target_text: Optional[str] = None) -> Dict[str, Any]:
    if not AZURE_AVAILABLE:
        raise ValueError("Azure SDKが利用できません")
    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    audio_config = speechsdk.audio.AudioConfig(filename=str(audio_path))
    if not target_text:
        target_text = azure_speech_to_text(audio_path, region, key)
    pronunciation_config = speechsdk.PronunciationAssessmentConfig(
        reference_text=target_text,
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue=True
    )
    pronunciation_config.enable_prosody_assessment()
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, language="en-US", audio_config=audio_config)
    pronunciation_config.apply_to(recognizer)
    result = recognizer.recognize_once()
    if result.reason == speechsdk.ResultReason.NoMatch:
        raise ValueError("No speech could be recognized.")
    pron_result = speechsdk.PronunciationAssessmentResult(result)
    return {
        "asr_text": result.text,
        "accuracy": pron_result.accuracy_score,
        "fluency": pron_result.fluency_score,
        "prosody": pron_result.prosody_score,
        "completeness": pron_result.completeness_score,
        "raw": json.loads(result.properties.get(speechsdk.PropertyId.SpeechServiceResponse_JsonResult))
    }

def openai_feedback(asr_text: str, target_text: str, azure_summary: Dict[str, Any], config: Dict[str, Any], task_type: str = "reading") -> str:
    _, _, openai_key = get_api_keys()
    if not openai_key:
        return "（注）OPENAI_API_KEY が未設定のため、AIフィードバックはスキップしました。"
    if not AZURE_AVAILABLE:
        return "（注）OpenAI SDKが利用できません。"
    client = OpenAI(api_key=openai_key)
    
    if task_type == "reading" and target_text:
        prompt = config["gpt_prompt_reading"].format(
            asr_text=asr_text,
            target_text=target_text,
            accuracy=azure_summary.get("accuracy"),
            fluency=azure_summary.get("fluency"),
            prosody=azure_summary.get("prosody")
        )
    else:
        prompt = config["gpt_prompt_speech"].format(
            asr_text=asr_text,
            accuracy=azure_summary.get("accuracy"),
            fluency=azure_summary.get("fluency"),
            prosody=azure_summary.get("prosody")
        )
    
    response = client.chat.completions.create(
        model=config["openai"]["model"],
        temperature=config["openai"]["temperature"],
        messages=[
            {"role": "system", "content": "You are a helpful assistant for Japanese university English speaking assessment."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def weighted_score(az: Dict[str, Any], content_org: int, vocab_gram: int, weights: Dict[str, int]) -> float:
    total = 0.0
    total += az.get("accuracy", 0) * (weights.get("pronunciation_accuracy", 0) / 100.0)
    total += az.get("fluency", 0) * (weights.get("fluency", 0) / 100.0)
    total += az.get("prosody", 0) * (weights.get("prosody", 0) / 100.0)
    total += content_org * (weights.get("content_organization", 0) / 100.0)
    total += vocab_gram * (weights.get("vocabulary_grammar", 0) / 100.0)
    return round(total, 1)

def band_from_score(score: float, bands: Dict[str, Dict[str, Any]]) -> str:
    best = "D"
    best_min = -1
    for b, cfg in bands.items():
        if score >= cfg["min"] and cfg["min"] > best_min:
            best, best_min = b, cfg["min"]
    return best

def map_to_cefr(score: float) -> str:
    if score >= 90: return "C1"
    elif score >= 80: return "B2"
    elif score >= 70: return "B1"
    elif score >= 55: return "A2"
    elif score >= 40: return "A1"
    else: return "Pre-A1"

def map_to_toefl_speaking(score: float) -> tuple:
    if score >= 90:
        toefl_score = 26 + int((score - 90) / 10 * 4)
        level = "Good (26-30)"
    elif score >= 80:
        toefl_score = 22 + int((score - 80) / 10 * 4)
        level = "Fair (18-25)"
    elif score >= 70:
        toefl_score = 18 + int((score - 70) / 10 * 4)
        level = "Fair (18-25)"
    elif score >= 55:
        toefl_score = 14 + int((score - 55) / 15 * 4)
        level = "Limited (10-17)"
    elif score >= 40:
        toefl_score = 10 + int((score - 40) / 15 * 4)
        level = "Limited (10-17)"
    else:
        toefl_score = max(0, int(score / 40 * 10))
        level = "Weak (0-9)"
    return min(toefl_score, 30), level

def map_to_ielts_speaking(score: float) -> float:
    if score >= 90:
        ielts_score = 8.0 + (score - 90) / 10 * 1.0
    elif score >= 80:
        ielts_score = 7.0 + (score - 80) / 10 * 1.0
    elif score >= 70:
        ielts_score = 6.0 + (score - 70) / 10 * 1.0
    elif score >= 60:
        ielts_score = 5.5 + (score - 60) / 10 * 0.5
    elif score >= 50:
        ielts_score = 5.0 + (score - 50) / 10 * 0.5
    elif score >= 40:
        ielts_score = 4.0 + (score - 40) / 10 * 1.0
    else:
        ielts_score = max(1.0, score / 40 * 4.0)
    return round(ielts_score * 2) / 2

def process_single_input(input_type, input_value, target_text: Optional[str] = None, task_type: str = "reading", config_path="config.yaml"):
    cfg = load_config(config_path)
    
    if task_type == "reading":
        weights = cfg["weights_reading"]
    else:
        weights = cfg["weights_speech"]
    
    key, region, _ = get_api_keys()
    
    if not key or not region:
        st.error("エラー: APIキーが設定されていません。Streamlit Cloud の Settings > Secrets で設定してください。")
        return None

    downloads_dir = Path(cfg.get("downloads_dir", "./downloads"))
    ensure_dir(downloads_dir)

    bands = {
        "A": {"min": 85, "label_ja": "A（到達目標を十分達成）"},
        "B": {"min": 70, "label_ja": "B（概ね達成）"},
        "C": {"min": 55, "label_ja": "C（一部達成）"},
        "D": {"min": 0,  "label_ja": "D（要改善）"}
    }

    try:
        if input_type == "youtube":
            audio_path = download_audio_from_youtube(input_value, downloads_dir)
        elif input_type == "google_drive":
            audio_path = download_from_google_drive(input_value, downloads_dir)
        elif input_type == "file":
            audio_path = extract_audio_from_file(input_value, downloads_dir)
        
        az = azure_pronunciation_assess(audio_path, region, key, target_text)
        
        if task_type == "reading":
            content_org = 70
            vocab_gram = 70
        else:
            content_org = 75
            vocab_gram = 75
        
        feedback = openai_feedback(az.get("asr_text", ""), target_text if target_text else "", az, cfg, task_type)
        total = weighted_score(az, content_org, vocab_gram, weights)
        band = band_from_score(total, bands)
        
        toefl_score, toefl_level = map_to_toefl_speaking(total)
        ielts_score = map_to_ielts_speaking(total)
        
        return {
            "score_total": total,
            "band": band,
            "cefr": map_to_cefr(total),
            "toefl_speaking": f"{toefl_score}/30 ({toefl_level})",
            "ielts_speaking": f"{ielts_score}/9.0",
            "accuracy": az["accuracy"],
            "fluency": az["fluency"],
            "prosody": az["prosody"],
            "completeness": az["completeness"],
            "asr_text": az["asr_text"],
            "comment": feedback,
            "task_type": task_type
        }
    except Exception as e:
        st.error(f"処理失敗: {str(e)}")
        return None

st.set_page_config(page_title="英語評価ツール（Azure版）", page_icon="🎯", layout="wide")

st.title("🎯 英語音読・スピーキング評価ツール")
st.caption("Azure Speech + OpenAI GPT版")

with st.expander("ℹ️ システム情報"):
    st.info("""
    **使用技術:**
    - 音声認識: Azure Speech Services
    - AI評価: OpenAI GPT-4o-mini
    
    **対応課題:**
    - 音読課題（目標テキストあり）
    - スピーチ課題（自由発話）
    
    **評価項目:**
    - 発音精度・流暢さ・プロソディ（音素レベル）
    - 国際基準スコア（CEFR/TOEFL/IELTS）
    """)

st.divider()

col1, col2 = st.columns([1, 3])
with col1:
    task_type = st.selectbox(
        "課題タイプ",
        ["音読課題", "スピーチ課題"],
        help="音読課題：目標テキストを読み上げ / スピーチ課題：自由発話"
    )
task_type_value = "reading" if task_type == "音読課題" else "speech"

with col2:
    if task_type == "音読課題":
        st.info("📖 発音精度を重視した評価（重み50%）")
    else:
        st.info("💬 内容構成も含めた総合評価（発音25%、内容25%）")

st.divider()

input_type = st.radio("入力方法を選択", ("MP3ファイル", "YouTubeリンク", "Google Driveリンク"), horizontal=True)

target_text = st.text_area(
    "目標テキスト（音読課題の場合）",
    placeholder="スピーチ課題の場合は空欄でOK",
    height=100
)

if input_type == "MP3ファイル":
    uploaded_file = st.file_uploader("MP3ファイルをアップロード", type=["mp3"])
    if uploaded_file and st.button("評価を実行", type="primary"):
        with st.spinner("処理中..."):
            result = process_single_input("file", uploaded_file, target_text if target_text else None, task_type_value)
            if result:
                st.success("評価完了！")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("総合スコア", f"{result['score_total']}")
                with col2:
                    st.metric("バンド", result['band'])
                with col3:
                    st.metric("CEFR", result['cefr'])
                with col4:
                    st.metric("TOEFL", result['toefl_speaking'].split('/')[0])
                
                st.divider()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("発音精度", f"{result['accuracy']}/100")
                with col2:
                    st.metric("流暢さ", f"{result['fluency']}/100")
                with col3:
                    st.metric("プロソディ", f"{result['prosody']}/100")
                
                st.divider()
                
                with st.expander("📝 書き起こしテキスト", expanded=True):
                    st.text(result.get('asr_text', ''))
                
                with st.expander("💬 フィードバック", expanded=True):
                    st.write(result['comment'])

elif input_type == "YouTubeリンク":
    url = st.text_input("YouTube限定公開リンク")
    if url and st.button("評価を実行", type="primary"):
        with st.spinner("処理中..."):
            result = process_single_input("youtube", url, target_text if target_text else None, task_type_value)
            if result:
                st.success("評価完了！")
                st.json(result)

elif input_type == "Google Driveリンク":
    url = st.text_input("Google Drive共有リンク（MP3のみ）")
    if url and st.button("評価を実行", type="primary"):
        with st.spinner("処理中..."):
            result = process_single_input("google_drive", url, target_text if target_text else None, task_type_value)
            if result:
                st.success("評価完了！")
                st.json(result)

st.divider()
st.caption("© 2025 English Assessment System | Azure Speech + OpenAI GPT")
