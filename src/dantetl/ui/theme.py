"""Shared desktop styling; overlay painting stays independent."""

STYLE = """
            QMainWindow, QWidget { background: #111923; color: #e7edf5; font: 10pt 'Segoe UI'; }
            QLabel#heading { font-size: 23pt; font-weight: 600; }
            QLabel#sectionTitle { font-size: 12pt; font-weight: 600; }
            QLabel#muted { color: #9baabc; font-size: 9pt; }
            QLabel#eyebrow { color: #92a4b7; font-size: 8pt; font-weight: 600; }
            QPushButton, QComboBox { background: #243142; border: 1px solid #3c4c60;
                border-radius: 6px; padding: 9px 12px; }
            QPushButton:hover { background: #304259; }
            QPushButton#favoriteWindow {
                color: #aab8c8; font: 14pt 'Segoe UI Symbol'; padding: 4px; }
            QPushButton#favoriteWindow:checked { color: #ffd35c; }
            QPushButton#favoriteWindow:disabled { color: #64748b; }
            QPushButton:disabled, QComboBox:disabled { color: #64748b; }
            QPushButton#primary { background: #257c73; border-color: #359c90; }
            QPushButton#primary:hover { background: #2d9185; }
            QPushButton#captureButton { background: #257c73; border-color: #359c90; }
            QPushButton#captureButton:hover { background: #2d9185; }
            QPushButton#captureButton[tone="recording"] {
                background: #a33343; border-color: #cf5769; }
            QPushButton#captureButton[tone="recording"]:hover { background: #b93c4e; }
            QPushButton#captureButton:disabled { color: #bdc5d0; }
            QTextBrowser { background: #172230; border: 1px solid #304259;
                border-radius: 6px; padding: 12px; }
            QComboBox QAbstractItemView { background: #1d2938;
                selection-background-color: #257c73; }
            QTableWidget { background: #172230; gridline-color: #304259;
                selection-background-color: #31544f; }
            QHeaderView::section { background: #243142; padding: 6px; border: 0; }
            QTableWidget#cloudModels::item { padding: 5px; border: 0; }
            QTableWidget#cloudModels::item:selected { background: #274746; color: #e7edf5; }
            QSlider::groove:horizontal { height: 4px; background: #35475c; }
            QSlider::handle:horizontal { background: #5eead4; width: 12px; margin: -4px 0;
                border-radius: 6px; }
            QProgressBar { border: 1px solid #3c4c60; border-radius: 4px;
                background: #172230; min-height: 18px; text-align: center; }
            QProgressBar::chunk { background: #257c73; border-radius: 3px; }

    QLineEdit#ollamaAddress { background: #172230; border: 1px solid #3c4c60;
        border-radius: 6px; padding: 9px 10px; }
    QLineEdit#ollamaAddress:focus { border-color: #5eead4; }
    QListWidget#ollamaModels { background: #172230; border: 1px solid #344557;
        border-radius: 6px; outline: 0; padding: 2px; }
    QListWidget#ollamaModels::item { padding: 0 8px; }
    QListWidget#ollamaModels::indicator:unchecked { background: #172230; border: 1px solid #8b9caf;
        border-radius: 2px; }
    QListWidget#ollamaModels::item:hover { background: #213142; }
    QListWidget#ollamaModels::item:selected { background: #274746; }
    QToolButton { color: #e7edf5; border: 1px solid transparent; padding: 6px; }
    QToolButton:hover { background: #243142; border-radius: 5px; }
    QToolButton:focus, QPushButton:focus, QComboBox:focus, QSpinBox:focus {
        border: 1px solid #5eead4; }
    QToolButton#info { color: #9fb7cb; font-size: 13pt; }
    QTabWidget::pane { border: 1px solid #344557; border-radius: 5px; }
    QTabBar::tab { background: #172230; padding: 10px 14px; color: #aabacc; }
    QTabBar::tab:selected { background: #243142; color: #e7edf5; }
    QSpinBox { padding: 7px; background: #243142; border: 1px solid #3c4c60; border-radius: 5px; }
    QSpinBox:disabled { color: #64748b; }
    QToolTip { background: #243142; color: #e7edf5; border: 1px solid #5eead4; padding: 7px; }
    QWidget#statusBanner { border-radius: 6px; }
    QWidget#statusBanner[tone="warning"] { background: #30291c; border: 1px solid #b58b3c; }
    QWidget#statusBanner[tone="error"] { background: #382127; border: 1px solid #c87580; }
    QWidget#statusBanner[tone="info"] { background: #1b2b39; border: 1px solid #557d9b; }
    QWidget#statusBanner QLabel { background: transparent; }
    QLabel#modelBadge { padding: 9px 12px; border-radius: 6px; font-weight: 600; }
    QLabel#modelBadge[tone="success"] { background: #183b32; color: #82e6bd;
        border: 1px solid #3ca983; }
    QLabel#modelBadge[tone="warning"] { background: #30291c; color: #f0ca7c;
        border: 1px solid #b58b3c; }
    QLabel#modelBadge[tone="info"] { background: #1b2b39; color: #a9c7df;
        border: 1px solid #557d9b; }
    QLabel#modelRating { font-weight: 600; }
    QLabel#modelRating[tone="success"] { color: #82e6bd; }
    QLabel#modelRating[tone="warning"] { color: #f0ca7c; }
    QLabel#modelRating[tone="error"] { color: #f09b9b; }
    QWidget#diagnosticCard { background: #172230; border: 1px solid #304259;
        border-radius: 7px; }
    QWidget#diagnosticCard QLabel { background: transparent; border: 0; }
    QLabel#diagnosticValue { font-size: 9pt; }
    QWidget#controlCard { background: #172230; border: 1px solid #304259;
        border-radius: 8px; }
    QWidget#controlCard QLabel, QWidget#controlCard QCheckBox {
        background: transparent; border: 0; }
    QWidget#routingSurface, QWidget#controlCard QToolButton { background: transparent; }
"""
