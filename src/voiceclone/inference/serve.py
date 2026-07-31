"""Deployed inference: a Modal class that keeps a fine-tuned checkpoint warm across
requests, exposed both as a programmatic `.remote()` method and as a small web
demo (Gradio, mounted on FastAPI) at a shareable Modal URL.

`experiment_name` is a `modal.parameter()` rather than hardcoded, so
`InferenceService(experiment_name="cfm_only")` and `InferenceService(experiment_name="full_ft_default")`
are independently deployable/callable — handy for the demo showing multiple
experiment checkpoints side by side (see plan Section 8, "show both 'fine-tuned' and
'fine-tuned + prompt' quality").
"""

import tempfile
from pathlib import Path

import modal

from voiceclone.config import Paths, get_settings
from voiceclone.data.manifest import pick_enrollment_utterance
from voiceclone.logging_utils import get_logger
from voiceclone.modal_app import GPU_CHEAP_INFERENCE, STANDARD_VOLUMES, app, cosyvoice_image

logger = get_logger(__name__)

settings = get_settings()


@app.cls(
    image=cosyvoice_image,
    volumes=STANDARD_VOLUMES,
    gpu=GPU_CHEAP_INFERENCE,
    scaledown_window=300,
)
class InferenceService:
    experiment_name: str = modal.parameter(default="full_ft_default")
    speaker_prefix: str = modal.parameter(default=settings.speaker_id)

    @modal.enter()
    def load(self) -> None:
        from voiceclone.inference.engine import DEFAULT_SPK_ID, VoiceCloneEngine

        checkpoint_dir = Paths.CHECKPOINT_DIR / self.speaker_prefix / self.experiment_name / "inference_ready"
        logger.info("loading checkpoint %s", checkpoint_dir)
        self.engine = VoiceCloneEngine(str(checkpoint_dir))

        train_dir = Paths.DATASET_DIR / self.speaker_prefix / "train"
        _utt, text, wav_path = pick_enrollment_utterance(train_dir)
        self.engine.enroll(text, wav_path, spk_id=DEFAULT_SPK_ID)

    @modal.method()
    def synthesize(self, text: str, speed: float = 1.0) -> bytes:
        from voiceclone.inference.engine import DEFAULT_SPK_ID

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "out.wav"
            self.engine.synthesize_to_file(text, out_path, spk_id=DEFAULT_SPK_ID, speed=speed)
            return out_path.read_bytes()

    @modal.asgi_app()
    def web(self):
        import gradio as gr
        from fastapi import FastAPI, Response
        from gradio.routes import mount_gradio_app
        from pydantic import BaseModel

        fastapi_app = FastAPI(title="voiceclone: CosyVoice3 personal voice demo")

        class SynthesizeRequest(BaseModel):
            text: str
            speed: float = 1.0

        @fastapi_app.post("/synthesize")
        def synthesize_endpoint(req: SynthesizeRequest) -> Response:
            wav_bytes = self.synthesize.local(req.text, req.speed)
            return Response(content=wav_bytes, media_type="audio/wav")

        def gradio_synthesize(text: str, speed: float) -> str:
            wav_bytes = self.synthesize.local(text, speed)
            tmp_path = "/tmp/voiceclone_demo_output.wav"
            with open(tmp_path, "wb") as f:
                f.write(wav_bytes)
            return tmp_path

        demo = gr.Interface(
            fn=gradio_synthesize,
            inputs=[
                gr.Textbox(label="Text to speak", lines=3, placeholder="Type something for the cloned voice to say..."),
                gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed"),
            ],
            outputs=gr.Audio(label="Generated speech", type="filepath"),
            title="Personal voice clone (CosyVoice3, fine-tuned)",
            description=f"Experiment: {self.experiment_name}",
        )
        return mount_gradio_app(fastapi_app, demo, path="/")
