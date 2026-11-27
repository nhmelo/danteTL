"""Compact, accessible model tradeoffs shared by settings and first-run setup."""

from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from dantetl.application.model_advice import ADVICE, COMPARISON_NOTE
from dantetl.i18n import mark, tr
from dantetl.ui.controls import InfoButton, set_tone
from dantetl.ui.localization import localized


def file_size(size: int) -> str:
    return (
        tr("{value:.2f} GB", value=size / 1000000000.0)
        if size >= 1e9
        else tr("{value:.0f} MB", value=size / 1000000.0)
    )


class ModelSummary(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.ratings = QWidget()
        grid = QGridLayout(self.ratings)
        grid.setContentsMargins(0, 0, 0, 0)
        self.values: list[QLabel] = []
        for index, name in enumerate(
            (mark("Dialogue"), mark("Game terms"), mark("Speed"), mark("Memory use"))
        ):
            row, column = index // 2, (index % 2) * 2
            grid.addWidget(localized(QLabel(), text=name), row, column)
            value = QLabel()
            value.setObjectName("modelRating")
            self.values.append(value)
            grid.addWidget(value, row, column + 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        layout.addWidget(self.ratings)
        capabilities = QHBoxLayout()
        self.cpu, self.gpu, self.context = QLabel(), QLabel(), QLabel()
        for label in (self.cpu, self.gpu, self.context):
            label.setWordWrap(True)
            capabilities.addWidget(label)
        layout.addLayout(capabilities)
        self.size_label = QLabel()
        self.size_label.setObjectName("muted")
        self.size_label.setWordWrap(True)
        footer = QHBoxLayout()
        footer.addWidget(self.size_label, 1)
        footer.addWidget(
            InfoButton(
                mark("Model ratings"),
                lambda: (
                    tr(COMPARISON_NOTE)
                    + tr(
                        " Speed assumes GPU for language models. "
                        "Capability checks indicate support, not an installed runtime. "
                        "Download size is not RAM/VRAM usage."
                    )
                ),
            )
        )
        layout.addLayout(footer)
        localized(self, toolTip=COMPARISON_NOTE)

    def set_model(
        self, key: str, size: int, language_model: bool, *, imported: bool = False
    ) -> None:
        advice = None if imported else ADVICE.get(key)
        self.ratings.setVisible(advice is not None)
        if advice:
            for label, rating in zip(
                self.values,
                (advice.dialogue, advice.terms, advice.speed, advice.memory),
                strict=True,
            ):
                label.setText(tr(rating.text))
                label.setToolTip(tr(rating.detail))
                set_tone(label, rating.tone)
        self.cpu.setText(tr("✓ CPU support"))
        self.gpu.setText(tr("✓ GPU support") if language_model else tr("— CPU only"))
        self.context.setText(tr("✓ Dialogue context") if language_model else tr("✕ No context"))
        self.gpu.setToolTip(tr("NVIDIA CUDA or Vulkan; optional runtime setup is separate."))
        self.cpu.setToolTip(tr("CPU execution is supported; larger models can be slow."))
        size_text = file_size(size)
        self.size_label.setText(
            (
                tr("Local file: {size}", size=size_text)
                if imported
                else tr("Download: {size}", size=size_text)
            )
            + (tr(" · Not rated") if not advice else "")
        )
        self.setVisible(bool(size))
