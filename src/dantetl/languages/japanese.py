"""Japanese script, word boundaries and game-text evidence."""

import re

from dantetl.languages.source import DashQuotes, ListSyntax, SourceLanguage

# Kana and kanji. Sound marks, the middle dot and the long-vowel mark, including halfwidth forms,
# belong to words but are not letter evidence alone; OCR reads icons and bars as them.
_SCRIPT = re.compile(
    r"[\u3041-\u3096\u309d-\u309f\u30a1-\u30fa\u30fd-\u30ff\u3400-\u9fff\uff66-\uff6f\uff71-\uff9d]"
)
_READING = re.compile(r"(?:\([ぁ-ゖァ-ヺー・ ]+\)|（[ぁ-ゖァ-ヺー・ ]+）)$")
_SET = re.compile(r"^[0-9０-９]+セット[:：]")
_OPERATION = re.compile(
    r"^[一-龯々ァ-ヺーA-Za-z0-9０-９]{0,16}(?:発動|起動|実行|使用|終了|解除)(?:中|時|後|前|間)(?:には|に|は)?[、，,]"
)
_PREDICATE = re.compile(r"を[^、，,。！？!?「」『』\s]{1,12}[うくぐすつぬぶむる]。$")
_DASH_QUOTES = DashQuotes(
    (("「", "」"), ("『", "』")), re.compile(r"^(?:-{2,}|[―—–]+)\s*(?=[「『/])")
)


class Japanese(SourceLanguage):
    __slots__ = ()
    trim_edge_icons = True
    extra_edge_marks = frozenset("「」『』【】〈〉《》。、・")
    tracking_brackets = SourceLanguage.tracking_brackets + "「」『』【】"
    word_separator = ""
    reveal_marks = "。、！？!?」』"
    terminal = tuple("。！？!?")
    endings = (*terminal, "」", "』")
    commas = tuple("、，,")
    label_punctuation = "。！？!?、，"
    speaker_delimiters = SourceLanguage.speaker_delimiters + "【】"
    lists = ListSyntax.from_punctuation(
        label_stops=label_punctuation + "「『", extra_bullets="・", extra_number_stops="、"
    )
    period = "。"
    prose_punctuation = "、，。"
    opening_punctuation = "、，。！？!?,…"
    prose_script = re.compile(r"[ぁ-ゖ]")
    short_row_cue = re.compile(r"。|^[はをにが]")
    ellipsis_clause_mark = "、"
    quote_start = re.compile(r"^(?:[―—–−]{1,3}|-{1,3}\s*[「『]|[「『])")
    dash_quotes = _DASH_QUOTES
    compact_counter_label = re.compile(r"[ァ-ヺー一-鿿々A-ZＡ-Ｚ]+")
    menu_labels = frozenset(
        [
            "詳細",
            "説明",
            "効果",
            "スキル",
            "秘技",
            "天賦",
            "アイテム",
            "装備",
            "能力",
            "所持",
            "確認",
            "戻る",
            "キャンセル",
            "おすすめ",
            "強化",
            "消費",
            "使用効果",
            "基本情報",
        ]
    )
    value_introductions = ("さらに", "更に", "最大で", "上限は")
    percentage_lead = re.compile(r"(?:[がはをもに]?(?:さらに|更に)|[がはをもに])")
    percentage_label_terms = ("による", "に対する")

    @property
    def standalone_menu_labels(self) -> frozenset[str]:
        return self.menu_labels | {"設定", "保存", "セーブ", "ロード", "終了"}

    def contains_text(self, text: str) -> bool:
        return _SCRIPT.search(text) is not None

    def needs_space(self, left: str, right: str) -> bool:
        return left.isascii() and left.isalnum() and right.isascii() and right.isalnum()

    def subtitle_continuation(self, first: str, tail: str) -> bool:
        # A centered final row can complete a split katakana word, conjugation or quoted term.
        if "ァ" <= first[-1] <= "ヺ" and tail[0] in "ァィゥェォャュョ":
            return True
        if first.endswith("っ") and tail[0] in "たて":
            return True
        if first[-1] not in "のはがをにとでへもて" and not self.internal_prose(first):
            return False
        for opener, closer in _DASH_QUOTES.pairs:
            term, mark, suffix = tail.partition(closer)
            if mark and suffix in ("だ", "です", "だった", "でした") and self.contains_text(term):
                return tail[0] == opener or not self.quote_start.match(tail)
        return False

    def section_heading(self, text: str) -> bool:
        return _SET.match(text) is not None

    def sample_effect_lettering(self, text: str) -> bool:
        return text.rstrip().endswith("。") and "を" in text

    def separate_effects(self, first: str, second: str) -> bool:
        return bool(
            _OPERATION.match(second) and _PREDICATE.search(first) and _PREDICATE.search(second)
        )

    def speaker_base(self, text: str) -> str:
        return _READING.sub("", text).strip()
