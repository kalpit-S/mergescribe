"""
Settings window, in native AppKit.

Laid out the way System Settings is: an icon sidebar, a large section title,
and rounded groups of rows with the label on the left and the control on the
right. Changes save as they are made, so there is no Save button to forget.

Runs in-process on the main thread, since rumps already owns the
NSApplication. The file layer lives in settings_store.py; this module is
presentation plus a table binding each control to the settings it writes.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import objc
from AppKit import (
    NSApplication, NSApplicationActivationPolicyRegular, NSBackingStoreBuffered,
    NSBezierPath, NSBox, NSButton, NSColor, NSFont, NSImage, NSImageView,
    NSMakeRect, NSMakeSize, NSMenu, NSMenuItem, NSPopUpButton, NSScrollView,
    NSSecureTextField, NSStackView, NSSwitch, NSTableCellView, NSTableColumn,
    NSTableView, NSTextField, NSTextView, NSView, NSVisualEffectView, NSWindow,
    NSWindowStyleMaskClosable, NSWindowStyleMaskFullSizeContentView,
    NSWindowStyleMaskMiniaturizable, NSWindowStyleMaskResizable,
    NSWindowStyleMaskTitled,
)
from Foundation import NSIndexSet, NSInsetRect, NSObject
from PyObjCTools import AppHelper

from ..validate import KeyValidator, ValidationResult
from .settings_store import (
    KNOWN_CORRECTION_MODELS, KNOWN_STT_MODELS, _DEFAULT_OR_CORRECTION_MODEL,
    _parse_model_ids, get_available_mics, get_routing_status, load_env_keys,
    load_settings, remove_settings, save_env_keys, save_settings,
)

WINDOW_SIZE = (780.0, 620.0)
MIN_SIZE = (680.0, 480.0)
SIDEBAR_WIDTH = 200.0
INSET = 28.0
ROW_INSETS = (9.0, 12.0, 9.0, 12.0)     # top, left, bottom, right
CARD_RADIUS = 9.0
FOOTER_WIDTH = 470.0

SECTIONS = [
    ("Recording", "waveform"),
    ("Models", "cpu"),
    ("Instructions", "text.quote"),
    ("Advanced", "slider.horizontal.3"),
]

TRIGGER_KEYS = [
    ("cmd_r", "Right Command"),
    ("alt_r", "Right Option"),
    ("alt_l", "Left Option"),
    ("ctrl_r", "Right Control"),
    ("f17", "F17"),
    ("f18", "F18"),
]

REASONING_EFFORTS = [
    ("", "Auto"), ("none", "None"), ("minimal", "Minimal"), ("low", "Low"),
    ("medium", "Medium"), ("high", "High"), ("xhigh", "XHigh"),
]

_CUSTOM = "__custom__"
_REMOVE = object()   # sentinel: delete this key from settings.json

# AppKit enum values, spelled out because PyObjC doesn't re-export them all.
_HORIZONTAL, _VERTICAL = 0, 1
_ALIGN_LEFT, _ALIGN_CENTER_Y = 1, 10
_DISTRIBUTE_FILL = 0
_BOX_SEPARATOR = 2
_SOURCE_LIST = 3
_MATERIAL_SIDEBAR, _BEHIND_WINDOW, _ALWAYS_ACTIVE = 7, 0, 1
_TRUNCATE_TAIL = 4
_TITLE_HIDDEN = 1
_SMALL = 1
_HUG_LOW, _HUG_HIGH, _RESIST_LOW = 1.0, 750.0, 250.0


# --------------------------------------------------------------- primitives

def _label(text: str, *, size: float = 13.0, bold: bool = False,
           secondary: bool = False, wraps: bool = False) -> NSTextField:
    field = (NSTextField.wrappingLabelWithString_(text) if wraps
             else NSTextField.labelWithString_(text))
    field.setFont_(NSFont.boldSystemFontOfSize_(size) if bold
                   else NSFont.systemFontOfSize_(size))
    if secondary:
        field.setTextColor_(NSColor.secondaryLabelColor())
    if wraps:
        field.setPreferredMaxLayoutWidth_(FOOTER_WIDTH)
    else:
        field.setLineBreakMode_(_TRUNCATE_TAIL)
    return field


def _fixed_width(view: NSView, width: float) -> NSView:
    view.setTranslatesAutoresizingMaskIntoConstraints_(False)
    view.widthAnchor().constraintEqualToConstant_(width).setActive_(True)
    return view


def _pin(child: NSView, parent: NSView, insets=(0.0, 0.0, 0.0, 0.0)) -> None:
    top, left, bottom, right = insets
    child.setTranslatesAutoresizingMaskIntoConstraints_(False)
    child.topAnchor().constraintEqualToAnchor_constant_(parent.topAnchor(), top).setActive_(True)
    child.leadingAnchor().constraintEqualToAnchor_constant_(
        parent.leadingAnchor(), left).setActive_(True)
    child.bottomAnchor().constraintEqualToAnchor_constant_(
        parent.bottomAnchor(), -bottom).setActive_(True)
    child.trailingAnchor().constraintEqualToAnchor_constant_(
        parent.trailingAnchor(), -right).setActive_(True)


def _vstack(spacing: float) -> NSStackView:
    stack = NSStackView.alloc().init()
    stack.setOrientation_(_VERTICAL)
    stack.setAlignment_(_ALIGN_LEFT)
    stack.setSpacing_(spacing)
    stack.setTranslatesAutoresizingMaskIntoConstraints_(False)
    return stack


def _set_detail(label: NSTextField, text: str) -> None:
    label.setStringValue_(text)
    label.setHidden_(not text)


def _clamped(raw: Any, lo: float, hi: float) -> Optional[float]:
    try:
        return max(lo, min(float(str(raw).strip()), hi))
    except (TypeError, ValueError):
        return None


def _trim(value: float, places: int = 2) -> str:
    """
    Show a number without trailing decimal zeros: 2.50 -> "2.5", 450 -> "450".

    Only strips when there is a decimal point. Stripping unconditionally turned
    450 into "45", and tabbing through the field then re-parsed that, clamped it
    to the minimum and saved a different value than the one on disk.
    """
    text = f"{value:.{places}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _lines(value: str) -> List[str]:
    return [line.strip() for line in (value or "").splitlines() if line.strip()]


def _words(value: str) -> List[str]:
    return _parse_model_ids((value or "").replace(" ", "\n"))


class _Card(NSView):
    """The rounded, faintly tinted container System Settings groups rows in."""

    def isFlipped(self):
        return True

    def drawRect_(self, rect):
        # Dynamic colours resolve at draw time, so this follows light/dark mode.
        path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSInsetRect(self.bounds(), 0.5, 0.5), CARD_RADIUS, CARD_RADIUS)
        NSColor.labelColor().colorWithAlphaComponent_(0.045).setFill()
        path.fill()
        NSColor.separatorColor().setStroke()
        path.setLineWidth_(1.0)
        path.stroke()


# ------------------------------------------------------------------- window

class SettingsWindowController(NSObject):
    """
    Owns the settings window. Main thread only.

    Every editable control is registered in `_bindings` with a function that
    returns the settings it should write, and the three AppKit change callbacks
    all funnel into `_commit_from`. That keeps each setting's read, write and
    validation in one place next to the control that edits it.
    """

    def init(self):
        self = objc.super(SettingsWindowController, self).init()
        if self is None:
            return None
        self.validator = KeyValidator()
        self._bindings: Dict[int, Callable[[], Dict[str, Any]]] = {}
        self._popup_values: Dict[int, List[Optional[str]]] = {}
        self._separator_above: Dict[int, NSView] = {}
        self._restore_policy: Optional[int] = None
        self._section_views: Dict[int, NSView] = {}
        self._load()
        self._build_window()
        return self

    @objc.python_method
    def _load(self) -> None:
        self.settings = load_settings()
        self.env_keys = load_env_keys()

    # -- chrome ------------------------------------------------------------

    @objc.python_method
    def _build_window(self) -> None:
        style = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
                 | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable
                 | NSWindowStyleMaskFullSizeContentView)
        width, height = WINDOW_SIZE
        window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, width, height), style, NSBackingStoreBuffered, False)
        window.setTitle_("MergeScribe Settings")
        window.setTitleVisibility_(_TITLE_HIDDEN)
        window.setTitlebarAppearsTransparent_(True)
        window.setContentMinSize_(NSMakeSize(*MIN_SIZE))
        window.setReleasedWhenClosed_(False)
        window.setDelegate_(self)
        window.center()
        self.window = window

        content = window.contentView()
        sidebar, detail = self._build_sidebar(), self._build_detail()
        content.addSubview_(sidebar)
        content.addSubview_(detail)
        for view in (sidebar, detail):
            view.setTranslatesAutoresizingMaskIntoConstraints_(False)
            view.topAnchor().constraintEqualToAnchor_(content.topAnchor()).setActive_(True)
            view.bottomAnchor().constraintEqualToAnchor_(content.bottomAnchor()).setActive_(True)
        sidebar.leadingAnchor().constraintEqualToAnchor_(content.leadingAnchor()).setActive_(True)
        sidebar.widthAnchor().constraintEqualToConstant_(SIDEBAR_WIDTH).setActive_(True)
        detail.leadingAnchor().constraintEqualToAnchor_(sidebar.trailingAnchor()).setActive_(True)
        detail.trailingAnchor().constraintEqualToAnchor_(content.trailingAnchor()).setActive_(True)

        # Selecting fires the delegate, which needs the detail pane to exist.
        self.table.selectRowIndexes_byExtendingSelection_(NSIndexSet.indexSetWithIndex_(0), False)
        self._show(0)

    @objc.python_method
    def _build_sidebar(self) -> NSView:
        effect = NSVisualEffectView.alloc().init()
        effect.setMaterial_(_MATERIAL_SIDEBAR)
        effect.setBlendingMode_(_BEHIND_WINDOW)
        effect.setState_(_ALWAYS_ACTIVE)

        table = NSTableView.alloc().init()
        table.addTableColumn_(NSTableColumn.alloc().initWithIdentifier_("section"))
        table.setHeaderView_(None)
        table.setStyle_(_SOURCE_LIST)
        table.setRowHeight_(30.0)
        table.setBackgroundColor_(NSColor.clearColor())
        table.setDataSource_(self)
        table.setDelegate_(self)
        self.table = table

        scroll = NSScrollView.alloc().init()
        scroll.setDocumentView_(table)
        scroll.setDrawsBackground_(False)
        scroll.setAutomaticallyAdjustsContentInsets_(False)
        effect.addSubview_(scroll)
        _pin(scroll, effect, (52.0, 10.0, 0.0, 10.0))   # clear the traffic lights
        return effect

    @objc.python_method
    def _build_detail(self) -> NSView:
        self.scroll = NSScrollView.alloc().init()
        self.scroll.setHasVerticalScroller_(True)
        self.scroll.setDrawsBackground_(False)
        self.scroll.setAutomaticallyAdjustsContentInsets_(False)
        return self.scroll

    @objc.python_method
    def _show(self, index: int) -> None:
        if not 0 <= index < len(SECTIONS):
            return
        if index not in self._section_views:
            builder = (self._recording_section, self._models_section,
                       self._instructions_section, self._advanced_section)[index]
            self._section_views[index] = builder()
        view = self._section_views[index]
        self.scroll.setDocumentView_(view)
        clip = self.scroll.contentView()
        view.topAnchor().constraintEqualToAnchor_(clip.topAnchor()).setActive_(True)
        view.leadingAnchor().constraintEqualToAnchor_(clip.leadingAnchor()).setActive_(True)
        view.widthAnchor().constraintEqualToAnchor_(clip.widthAnchor()).setActive_(True)

    # -- sidebar data source -----------------------------------------------

    def numberOfRowsInTableView_(self, table) -> int:
        return len(SECTIONS)

    def tableView_viewForTableColumn_row_(self, table, column, row):
        cell = table.makeViewWithIdentifier_owner_("SidebarCell", self)
        if cell is None:
            cell = NSTableCellView.alloc().init()
            cell.setIdentifier_("SidebarCell")
            icon, text = NSImageView.alloc().init(), NSTextField.labelWithString_("")
            text.setFont_(NSFont.systemFontOfSize_(13.0))
            for view in (icon, text):
                view.setTranslatesAutoresizingMaskIntoConstraints_(False)
                cell.addSubview_(view)
                view.centerYAnchor().constraintEqualToAnchor_(cell.centerYAnchor()).setActive_(True)
            icon.leadingAnchor().constraintEqualToAnchor_constant_(
                cell.leadingAnchor(), 6.0).setActive_(True)
            icon.widthAnchor().constraintEqualToConstant_(18.0).setActive_(True)
            icon.heightAnchor().constraintEqualToConstant_(18.0).setActive_(True)
            text.leadingAnchor().constraintEqualToAnchor_constant_(
                icon.trailingAnchor(), 8.0).setActive_(True)
            cell.setImageView_(icon)
            cell.setTextField_(text)
        title, symbol = SECTIONS[row]
        cell.textField().setStringValue_(title)
        cell.imageView().setImage_(
            NSImage.imageWithSystemSymbolName_accessibilityDescription_(symbol, title))
        return cell

    def tableViewSelectionDidChange_(self, notification) -> None:
        self._show(int(self.table.selectedRow()))

    # -- composition ---------------------------------------------------------

    @objc.python_method
    def _row(self, title: str, control: Optional[NSView] = None,
             detail: str = "") -> Tuple[NSStackView, NSTextField]:
        """Title (and optional detail line) on the left, control on the right."""
        text = _vstack(1.0)
        heading = _label(title)
        heading.setContentCompressionResistancePriority_forOrientation_(_RESIST_LOW, _HORIZONTAL)
        note = _label("", size=11.0, secondary=True)
        _set_detail(note, detail)
        text.addArrangedSubview_(heading)
        text.addArrangedSubview_(note)
        text.setHuggingPriority_forOrientation_(_HUG_LOW, _HORIZONTAL)

        row = NSStackView.alloc().init()
        row.setOrientation_(_HORIZONTAL)
        row.setAlignment_(_ALIGN_CENTER_Y)
        row.setDistribution_(_DISTRIBUTE_FILL)
        row.setSpacing_(16.0)
        row.setEdgeInsets_(ROW_INSETS)
        row.setTranslatesAutoresizingMaskIntoConstraints_(False)
        row.addArrangedSubview_(text)
        if control is not None:
            control.setContentHuggingPriority_forOrientation_(_HUG_HIGH, _HORIZONTAL)
            row.addArrangedSubview_(control)
        return row, note

    @objc.python_method
    def _text_area(self, value: str, height: float) -> Tuple[NSView, NSTextView]:
        """A borderless editor that sits inside a card."""
        scroll = NSTextView.scrollableTextView()
        text = scroll.documentView()
        text.setString_(value or "")
        text.setFont_(NSFont.systemFontOfSize_(13.0))
        text.setRichText_(False)
        # Substitutions would silently rewrite prompts that forbid those glyphs.
        text.setAutomaticQuoteSubstitutionEnabled_(False)
        text.setAutomaticDashSubstitutionEnabled_(False)
        text.setAutomaticTextReplacementEnabled_(False)
        text.setDrawsBackground_(False)
        text.setTextContainerInset_(NSMakeSize(0.0, 2.0))
        text.setDelegate_(self)
        scroll.setDrawsBackground_(False)
        scroll.setBorderType_(0)
        scroll.setAutohidesScrollers_(True)

        wrapper = NSView.alloc().init()
        wrapper.setTranslatesAutoresizingMaskIntoConstraints_(False)
        wrapper.addSubview_(scroll)
        _pin(scroll, wrapper, (8.0, 10.0, 8.0, 10.0))
        scroll.heightAnchor().constraintEqualToConstant_(height).setActive_(True)
        return wrapper, text

    @objc.python_method
    def _card(self, rows: List[NSView]) -> NSView:
        card = _Card.alloc().init()
        card.setTranslatesAutoresizingMaskIntoConstraints_(False)
        stack = _vstack(0.0)
        card.addSubview_(stack)
        _pin(stack, card)
        for index, row in enumerate(rows):
            if index:
                line = NSView.alloc().init()
                line.setTranslatesAutoresizingMaskIntoConstraints_(False)
                rule = NSBox.alloc().init()
                rule.setBoxType_(_BOX_SEPARATOR)
                line.addSubview_(rule)
                _pin(rule, line, (0.0, ROW_INSETS[1], 0.0, 0.0))
                line.heightAnchor().constraintEqualToConstant_(1.0).setActive_(True)
                stack.addArrangedSubview_(line)
                line.widthAnchor().constraintEqualToAnchor_(stack.widthAnchor()).setActive_(True)
                self._separator_above[objc.pyobjc_id(row)] = line
            stack.addArrangedSubview_(row)
            row.widthAnchor().constraintEqualToAnchor_(stack.widthAnchor()).setActive_(True)
        return card

    @objc.python_method
    def _section(self, title: str,
                 groups: List[Tuple[str, NSView, Union[str, NSTextField]]]) -> NSView:
        stack = _vstack(6.0)
        stack.setEdgeInsets_((40.0, INSET, 32.0, INSET))
        heading = _label(title, size=22.0, bold=True)
        stack.addArrangedSubview_(heading)
        stack.setCustomSpacing_afterView_(18.0, heading)
        for header, card, footer in groups:
            if header:
                caption = _label(header, bold=True)
                stack.addArrangedSubview_(caption)
            stack.addArrangedSubview_(card)
            card.widthAnchor().constraintEqualToAnchor_constant_(
                stack.widthAnchor(), -2 * INSET).setActive_(True)
            last = card
            if footer:
                note = footer if isinstance(footer, NSTextField) else _label(
                    footer, size=11.0, secondary=True, wraps=True)
                stack.addArrangedSubview_(note)
                stack.setCustomSpacing_afterView_(6.0, card)
                last = note
            stack.setCustomSpacing_afterView_(24.0, last)
        return stack

    @objc.python_method
    def _set_row_hidden(self, row: NSView, hidden: bool) -> None:
        row.setHidden_(hidden)
        line = self._separator_above.get(objc.pyobjc_id(row))
        if line is not None:
            line.setHidden_(hidden)

    # -- controls ------------------------------------------------------------

    @objc.python_method
    def _bind(self, control, produce: Callable[[], Dict[str, Any]]) -> None:
        self._bindings[objc.pyobjc_id(control)] = produce

    @objc.python_method
    def _switch(self, on: bool) -> NSSwitch:
        switch = NSSwitch.alloc().init()
        switch.setState_(1 if on else 0)
        switch.setControlSize_(_SMALL)
        switch.setTarget_(self)
        switch.setAction_("changed:")
        return switch

    @objc.python_method
    def _popup(self, options: List[Tuple[Optional[str], str]], selected: Optional[str],
               width: float) -> NSPopUpButton:
        """A pop-up menu; an option whose value is None becomes a separator."""
        popup = NSPopUpButton.alloc().initWithFrame_pullsDown_(NSMakeRect(0, 0, width, 24), False)
        values: List[Optional[str]] = []
        for value, title in options:
            if value is None:
                popup.menu().addItem_(NSMenuItem.separatorItem())
            else:
                popup.addItemWithTitle_(title)
            values.append(value)
        if selected in values:
            popup.selectItemAtIndex_(values.index(selected))
        popup.setTarget_(self)
        popup.setAction_("changed:")
        self._popup_values[objc.pyobjc_id(popup)] = values
        return _fixed_width(popup, width)

    @objc.python_method
    def _popup_value(self, popup: NSPopUpButton) -> Optional[str]:
        values = self._popup_values[objc.pyobjc_id(popup)]
        index = int(popup.indexOfSelectedItem())
        return values[index] if 0 <= index < len(values) else None

    @objc.python_method
    def _field(self, value: str, width: float, placeholder: str = "",
               secure: bool = False) -> NSTextField:
        field = (NSSecureTextField if secure else NSTextField).alloc().init()
        field.setStringValue_(value or "")
        field.setPlaceholderString_(placeholder)
        field.setDelegate_(self)
        return _fixed_width(field, width)

    @objc.python_method
    def _number(self, key: str, value: float, lo: float, hi: float, places: int = 2,
                scale: float = 1.0) -> NSTextField:
        """A numeric field that clamps what you type and shows what was kept."""
        field = self._field(_trim(value * scale, places), 72.0)
        field.setAlignment_(2)   # right-aligned, like System Settings

        def produce() -> Dict[str, Any]:
            parsed = _clamped(field.stringValue(), lo * scale, hi * scale)
            if parsed is None:
                field.setStringValue_(_trim(self.settings.get(key, value) * scale, places))
                return {}
            field.setStringValue_(_trim(parsed, places))
            return {key: parsed / scale}
        self._bind(field, produce)
        return field

    # -- change callbacks ----------------------------------------------------

    def changed_(self, sender) -> None:
        self._commit_from(sender)

    def controlTextDidEndEditing_(self, notification) -> None:
        self._commit_from(notification.object())

    def textDidEndEditing_(self, notification) -> None:
        self._commit_from(notification.object())

    @objc.python_method
    def _commit_from(self, control) -> None:
        produce = self._bindings.get(objc.pyobjc_id(control))
        if produce is None:
            return
        try:
            values = produce() or {}
            keep = {k: v for k, v in values.items() if v is not _REMOVE}
            drop = [k for k, v in values.items() if v is _REMOVE]
            if keep:
                save_settings(keep)
            if drop:
                remove_settings(drop)
            self.settings.update(keep)
            for key in drop:
                self.settings.pop(key, None)
        except Exception as e:
            print(f"[Settings] Could not save: {e}")

    # -- Recording -----------------------------------------------------------

    @objc.python_method
    def _recording_section(self) -> NSView:
        s = self.settings
        enabled = s.get("enabled_mics", s.get("ENABLED_INPUT_DEVICES", []))
        mic_switches: Dict[str, NSSwitch] = {}
        mic_rows: List[NSView] = []
        for mic in get_available_mics():
            switch = self._switch(mic in enabled)
            mic_switches[mic] = switch
            mic_rows.append(self._row(mic, switch)[0])
        for switch in mic_switches.values():
            self._bind(switch, lambda: {
                "enabled_mics": [m for m, sw in mic_switches.items() if sw.state()]})
        if not mic_rows:
            mic_rows.append(self._row("No microphones found")[0])

        trigger = self._popup(TRIGGER_KEYS, s.get("trigger_key", "alt_r"), 170.0)
        self._bind(trigger, lambda: {"trigger_key": self._popup_value(trigger)})
        tap = self._number("double_tap_threshold",
                           _clamped(s.get("double_tap_threshold", 0.45), 0.15, 1.0) or 0.45,
                           0.15, 1.0, places=0, scale=1000.0)

        def toggle(key: str, default: bool) -> NSSwitch:
            switch = self._switch(s.get(key, default))
            self._bind(switch, lambda: {key: bool(switch.state())})
            return switch

        return self._section("Recording", [
            ("Microphones", self._card(mic_rows), "Every enabled microphone is recorded in parallel."),
            ("Shortcut", self._card([
                self._row("Trigger key", trigger, "Hold to record, double-tap to toggle")[0],
                self._row("Double-tap window", tap, "Milliseconds")[0],
            ]), ""),
            ("While Recording", self._card([
                self._row("Recording HUD", toggle("hud_enabled", True),
                          "Mic level, live transcript and progress")[0],
                self._row("Space between dictations", toggle("space_between_dictations", True),
                          "When continuing in the same app")[0],
            ]), ""),
            ("Speech Detection", self._card([
                self._row("Split on silence", toggle("chunk_on_silence", True),
                          "Transcribe while you're still talking")[0],
                self._row("Adaptive threshold", toggle("adaptive_threshold_enabled", True),
                          "Follows the room's noise floor")[0],
                self._row("Split after silence",
                          self._number("silence_threshold",
                                       _clamped(s.get("silence_threshold", 1.2), 0.2, 10.0) or 1.2,
                                       0.2, 10.0), "Seconds")[0],
                self._row("Speech threshold",
                          self._number("speech_headroom_db",
                                       _clamped(s.get("speech_headroom_db", 12.0), 1.0, 30.0) or 12.0,
                                       1.0, 30.0, places=1), "Decibels above the noise floor")[0],
            ]), "Raise the threshold in noisy rooms; lower it if quiet speech gets missed."),
        ])

    # -- Models --------------------------------------------------------------

    @objc.python_method
    def _models_section(self) -> NSView:
        s = self.settings
        has_key = bool(self.env_keys.get("OPENROUTER_API_KEY"))

        self.api_key = self._field(self.env_keys.get("OPENROUTER_API_KEY", ""), 260.0,
                                   "sk-or-…", secure=True)
        key_row, self.key_status = self._row("API key", self.api_key)
        self._bind(self.api_key, self._commit_key)

        providers = {p.replace("_mlx", "").replace("_whisper", "")
                     for p in s.get("enabled_providers", ["parakeet"])}
        parakeet = self._switch("parakeet" in providers)
        self._bind(parakeet, lambda: {
            "enabled_providers": ["parakeet"] if parakeet.state() else []})

        chosen = set(s.get("openrouter_stt_models", []))
        known = {slug for slug, _ in KNOWN_STT_MODELS}
        self.stt_switches: Dict[str, Tuple[NSSwitch, NSTextField]] = {}
        stt_rows: List[NSView] = [self._row("Parakeet", parakeet, "Local, on this Mac")[0]]
        for slug, title in KNOWN_STT_MODELS:
            switch = self._switch(slug in chosen)
            row, note = self._row(title, switch)
            self.stt_switches[slug] = (switch, note)
            stt_rows.append(row)
        self.extra_stt = self._field(
            " ".join(m for m in s.get("openrouter_stt_models", []) if m not in known),
            240.0, "vendor/model, space separated")
        stt_rows.append(self._row("Other models", self.extra_stt)[0])

        def produce_stt() -> Dict[str, Any]:
            models = [slug for slug, (sw, _) in self.stt_switches.items() if sw.state()]
            models += [m for m in _words(self.extra_stt.stringValue()) if m not in models]
            return {"openrouter_stt_models": models}
        for switch, _ in self.stt_switches.values():
            self._bind(switch, produce_stt)
        self._bind(self.extra_stt, produce_stt)

        model = s.get("openrouter_correction_model", _DEFAULT_OR_CORRECTION_MODEL)
        known_models = [slug for slug, _ in KNOWN_CORRECTION_MODELS]
        options = list(KNOWN_CORRECTION_MODELS) + [(None, ""), (_CUSTOM, "Custom…")]
        self.model_popup = self._popup(options, model if model in known_models else _CUSTOM, 240.0)
        self.custom_model = self._field("" if model in known_models else model, 240.0,
                                        "vendor/model")
        self.custom_row = self._row("Model ID", self.custom_model)[0]
        self.effort = self._popup(REASONING_EFFORTS,
                                  s.get("openrouter_correction_reasoning_effort", ""), 130.0)
        self.provider_order = self._field(
            " ".join(s.get("openrouter_correction_provider_order", [])), 200.0, "e.g. together")
        fallbacks = self._switch(s.get("openrouter_correction_allow_fallbacks", True))
        self.routing_note = _label("", size=11.0, secondary=True, wraps=True)

        self._bind(self.model_popup, self._commit_model_choice)
        self._bind(self.custom_model, self._commit_custom_model)
        self._bind(self.effort, lambda: self._after_routing_change(
            {"openrouter_correction_reasoning_effort": self._popup_value(self.effort) or ""}))
        self._bind(self.provider_order, lambda: self._after_routing_change(
            {"openrouter_correction_provider_order": _words(self.provider_order.stringValue())}))
        self._bind(fallbacks, lambda: {
            "openrouter_correction_allow_fallbacks": bool(fallbacks.state())})

        section = self._section("Models", [
            ("OpenRouter", self._card([key_row]),
             "Used for cloud transcription and for correction."),
            ("Transcription", self._card(stt_rows),
             "Everything enabled runs in parallel; the correction model reconciles them."),
            ("Correction", self._card([
                self._row("Model", self.model_popup)[0],
                self.custom_row,
                self._row("Reasoning", self.effort)[0],
            ]), self.routing_note),
            ("Provider Routing", self._card([
                self._row("Provider order", self.provider_order,
                          "Preferred OpenRouter providers")[0],
                self._row("Allow fallbacks", fallbacks)[0],
            ]), ""),
        ])
        self._set_row_hidden(self.custom_row, self._popup_value(self.model_popup) != _CUSTOM)
        self._refresh_key_dependent(has_key)
        self._refresh_routing()
        self._validate_key()
        return section

    @objc.python_method
    def _selected_model(self) -> str:
        choice = self._popup_value(self.model_popup)
        if choice == _CUSTOM:
            return self.custom_model.stringValue().strip() or _DEFAULT_OR_CORRECTION_MODEL
        return choice or _DEFAULT_OR_CORRECTION_MODEL

    @objc.python_method
    def _commit_model_choice(self) -> Dict[str, Any]:
        custom = self._popup_value(self.model_popup) == _CUSTOM
        self._set_row_hidden(self.custom_row, not custom)
        if custom and not self.custom_model.stringValue().strip():
            self.window.makeFirstResponder_(self.custom_model)
            return {}   # nothing to save until an id is typed
        return self._after_routing_change({"openrouter_correction_model": self._selected_model()})

    @objc.python_method
    def _commit_custom_model(self) -> Dict[str, Any]:
        if not self.custom_model.stringValue().strip():
            return {}
        return self._after_routing_change({"openrouter_correction_model": self._selected_model()})

    @objc.python_method
    def _after_routing_change(self, values: Dict[str, Any]) -> Dict[str, Any]:
        self.settings.update(values)
        self._refresh_routing()
        return values

    @objc.python_method
    def _refresh_routing(self) -> None:
        self.routing_note.setStringValue_(get_routing_status(
            self.env_keys.get("OPENROUTER_API_KEY", ""),
            self._selected_model(),
            _words(self.provider_order.stringValue()),
            self._popup_value(self.effort) or "",
        ))

    @objc.python_method
    def _refresh_key_dependent(self, has_key: bool) -> None:
        for switch, note in self.stt_switches.values():
            switch.setEnabled_(has_key)
            _set_detail(note, "" if has_key else "Needs an OpenRouter key")
        self.extra_stt.setEnabled_(has_key)

    @objc.python_method
    def _commit_key(self) -> Dict[str, Any]:
        key = self.api_key.stringValue().strip()
        if key == (self.env_keys.get("OPENROUTER_API_KEY") or ""):
            return {}
        save_env_keys({"OPENROUTER_API_KEY": key})
        self.env_keys = load_env_keys()
        self._refresh_key_dependent(bool(key))
        self._refresh_routing()
        self._validate_key()
        return {}

    @objc.python_method
    def _validate_key(self) -> None:
        key = self.env_keys.get("OPENROUTER_API_KEY", "")
        if not key:
            _set_detail(self.key_status, "Not set")
            return
        _set_detail(self.key_status, "Checking…")
        self.key_status.setTextColor_(NSColor.secondaryLabelColor())
        # The validator answers on a worker thread; AppKit must be touched on main.
        self.validator.validate_openrouter(
            key, lambda result: AppHelper.callAfter(self._show_key_result, result))

    @objc.python_method
    def _show_key_result(self, result: ValidationResult) -> None:
        if result.valid:
            _set_detail(self.key_status, f"Connected · {result.latency_ms} ms")
            self.key_status.setTextColor_(NSColor.systemGreenColor())
        else:
            _set_detail(self.key_status, str(result.error or "Invalid key"))
            self.key_status.setTextColor_(NSColor.systemRedColor())

    # -- Instructions --------------------------------------------------------

    @objc.python_method
    def _instructions_section(self) -> NSView:
        s = self.settings
        about, about_text = self._text_area(s.get("custom_instructions", ""), 170.0)
        self._bind(about_text, lambda: {"custom_instructions": about_text.string()})

        routing = self._switch(s.get("field_routing_enabled", False))
        self._bind(routing, lambda: {"field_routing_enabled": bool(routing.state())})
        apps, apps_text = self._text_area("\n".join(s.get("routing_allowed_apps", [])), 64.0)
        self._bind(apps_text, lambda: {"routing_allowed_apps": _lines(apps_text.string())})
        prefs, prefs_text = self._text_area(s.get("routing_instructions", ""), 96.0)
        self._bind(prefs_text, lambda: {"routing_instructions": prefs_text.string().strip()})

        from ..vocabulary import learned_corrections
        learning = self._switch(s.get("learn_vocabulary", True))
        self._bind(learning, lambda: {"learn_vocabulary": bool(learning.state())})
        learned = [after for _, after, _ in learned_corrections()]
        learned_note = ("Learned so far: " + ", ".join(learned)) if learned else "Nothing learned yet."

        return self._section("Instructions", [
            ("About You", self._card([about]),
             "Who you are and the names you use. The model also sees the active app, "
             "so app-specific rules work here too."),
            ("Learning", self._card([
                self._row("Learn from corrections", learning,
                          "A word you correct in two separate dictations is shown to the model")[0],
            ]), learned_note),
            ("Output Routing", self._card([
                self._row("Route to the best field", routing, "Experimental")[0],
            ]), "When on, dictation can land in another app's text field instead of "
                "the one you're focused on."),
            ("Eligible Apps", self._card([apps]),
             "One app per line. Leave empty to allow every app on screen."),
            ("Routing Preferences", self._card([prefs]),
             "Where different kinds of dictation should go."),
        ])

    # -- Advanced ------------------------------------------------------------

    @objc.python_method
    def _advanced_section(self) -> NSView:
        from ..correct import DEFAULT_EDITING_PROMPT, DEFAULT_SYSTEM_CONTEXT

        def prompt_card(key: str, default: str, height: float) -> NSView:
            area, text = self._text_area(self.settings.get(key, default), height)
            # Persist only a genuine override, so the built-in default can evolve.
            self._bind(text, lambda: {
                key: _REMOVE if text.string().strip() == default.strip() else text.string()})
            reset = NSButton.buttonWithTitle_target_action_("Reset to Default", self, "reset:")
            reset.setControlSize_(_SMALL)
            self._resets[objc.pyobjc_id(reset)] = (key, default, text)
            footer = self._row("", reset)[0]
            return self._card([area, footer])

        self._resets: Dict[int, Tuple[str, str, NSTextView]] = {}
        return self._section("Advanced", [
            ("Correction Prompt",
             prompt_card("system_prompt", DEFAULT_SYSTEM_CONTEXT, 230.0),
             "How transcripts are cleaned up before they're typed."),
            ("Editing Prompt",
             prompt_card("editing_prompt", DEFAULT_EDITING_PROMPT, 80.0),
             "Used when you select text and speak an instruction."),
        ])

    def reset_(self, sender) -> None:
        key, default, text = self._resets[objc.pyobjc_id(sender)]
        text.setString_(default)
        remove_settings([key])
        self.settings.pop(key, None)

    # -- lifecycle -----------------------------------------------------------

    def windowWillClose_(self, notification) -> None:
        # Ending editing fires the delegates, so a half-typed field still saves.
        self.window.makeFirstResponder_(None)
        if self._restore_policy is not None:
            NSApplication.sharedApplication().setActivationPolicy_(self._restore_policy)
            self._restore_policy = None

    @objc.python_method
    def _ensure_edit_menu(self) -> None:
        """
        Give the app an Edit menu while settings is open.

        A menu-bar app has no main menu, and ⌘C, ⌘V and ⌘A are dispatched
        through Edit-menu key equivalents, so without one the text fields
        silently ignore copy, paste and select-all.
        """
        app = NSApplication.sharedApplication()
        main = app.mainMenu()
        if main is not None and main.itemWithTitle_("Edit") is not None:
            return
        if main is None:
            main = NSMenu.alloc().initWithTitle_("Main")
            app.setMainMenu_(main)
        if main.numberOfItems() == 0:
            app_item, app_menu = NSMenuItem.alloc().init(), NSMenu.alloc().initWithTitle_("MergeScribe")
            app_menu.addItemWithTitle_action_keyEquivalent_("Close Window", "performClose:", "w")
            app_item.setSubmenu_(app_menu)
            main.addItem_(app_item)
        edit_item, edit = NSMenuItem.alloc().init(), NSMenu.alloc().initWithTitle_("Edit")
        for entry in (("Undo", "undo:", "z"), ("Redo", "redo:", "Z"), None,
                      ("Cut", "cut:", "x"), ("Copy", "copy:", "c"),
                      ("Paste", "paste:", "v"), ("Select All", "selectAll:", "a")):
            if entry is None:
                edit.addItem_(NSMenuItem.separatorItem())
            else:
                edit.addItemWithTitle_action_keyEquivalent_(*entry)
        edit_item.setSubmenu_(edit)
        main.addItem_(edit_item)

    @objc.python_method
    def present(self) -> None:
        app = NSApplication.sharedApplication()
        if not self.window.isVisible():
            # Rebuild from disk so a reopened window never shows stale values.
            self._load()
            self._section_views.clear()
            self._bindings.clear()
            self._popup_values.clear()
            self._separator_above.clear()
            self._show(max(0, int(self.table.selectedRow())))
        if self._restore_policy is None:
            # An accessory app's windows can't become key, so text can't be
            # selected or typed into. Become a regular app while settings is up.
            self._restore_policy = int(app.activationPolicy())
            app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
        self._ensure_edit_menu()
        self.window.makeKeyAndOrderFront_(None)
        if hasattr(app, "activate"):
            app.activate()
        else:
            app.activateIgnoringOtherApps_(True)


_controller: Optional[SettingsWindowController] = None


def open_settings() -> None:
    """
    Show the settings window. Must be called on the main thread.

    The controller is cached so reopening reuses the window, and so Python
    does not collect it while AppKit still has the window on screen.
    """
    global _controller
    if _controller is None:
        _controller = SettingsWindowController.alloc().init()
    _controller.present()
