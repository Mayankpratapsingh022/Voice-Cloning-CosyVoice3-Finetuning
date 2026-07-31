"""Deployed inference: a FastAPI app (with a Gradio demo mounted on it) serving one
fine-tuned checkpoint. Runs on the RunPod pod, bound to 0.0.0.0:8000, which the pod
exposes publicly (see runpod_app.py's SERVE_PORT_LABEL and the pod's proxy URL,
printed by `voiceclone serve start`).

Unlike the earlier Modal version, this is a plain long-running process, not a
container lifecycle hook. `voiceclone remote serve` (see cli.py) runs this on the pod
over SSH with the checkpoint already loaded once at startup, not per-request.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from voiceclone.config import RemotePaths
from voiceclone.data.manifest import pick_enrollment_utterance
from voiceclone.logging_utils import get_logger

logger = get_logger(__name__)


def build_app(speaker_prefix: str, experiment_name: str):
    import gradio as gr
    from fastapi import FastAPI, Response
    from gradio.routes import mount_gradio_app
    from pydantic import BaseModel

    from voiceclone.inference.engine import DEFAULT_SPK_ID, VoiceCloneEngine

    checkpoint_dir = RemotePaths.CHECKPOINT_DIR / speaker_prefix / experiment_name / "inference_ready"
    logger.info("loading checkpoint %s", checkpoint_dir)
    engine = VoiceCloneEngine(str(checkpoint_dir))

    train_dir = RemotePaths.DATASET_DIR / speaker_prefix / "train"
    _utt, text, wav_path = pick_enrollment_utterance(train_dir)
    engine.enroll(text, wav_path, spk_id=DEFAULT_SPK_ID)

    def synthesize(text: str, speed: float = 1.0) -> bytes:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "out.wav"
            engine.synthesize_to_file(text, out_path, spk_id=DEFAULT_SPK_ID, speed=speed)
            return out_path.read_bytes()

    fastapi_app = FastAPI(title="voiceclone: CosyVoice3 personal voice demo")

    class SynthesizeRequest(BaseModel):
        text: str
        speed: float = 1.0

    @fastapi_app.post("/synthesize")
    def synthesize_endpoint(req: SynthesizeRequest) -> Response:
        return Response(content=synthesize(req.text, req.speed), media_type="audio/wav")

    def gradio_synthesize(text: str, speed: float) -> str:
        tmp_path = "/tmp/voiceclone_demo_output.wav"
        with open(tmp_path, "wb") as f:
            f.write(synthesize(text, speed))
        return tmp_path

    demo = gr.Interface(
        fn=gradio_synthesize,
        inputs=[
            gr.Textbox(label="Text to speak", lines=3, placeholder="Type something for the cloned voice to say..."),
            gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed"),
        ],
        outputs=gr.Audio(label="Generated speech", type="filepath"),
        title="Personal voice clone (CosyVoice3, fine-tuned)",
        description=f"Experiment: {experiment_name}",
    )
    return mount_gradio_app(fastapi_app, demo, path="/")


def run_server(speaker_prefix: str, experiment_name: str, port: int = 8000) -> None:
    import uvicorn

    app = build_app(speaker_prefix, experiment_name)
    uvicorn.run(app, host="0.0.0.0", port=port)  # noqa: S104 -- intentional: the pod's edge/proxy is what actually gates public access, not this bind
