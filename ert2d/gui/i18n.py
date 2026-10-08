"""Lightweight Turkish / English translation.

No Qt ``.ts`` tooling - just a dict and a callable.  Widgets connect to
:data:`translator` ``.language_changed`` and re-apply their strings in a
``retranslate()`` method.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

_STR: dict[str, dict[str, str]] = {
    # --- window / menus -------------------------------------------------
    "app.title": {"en": "ELRIS 2D", "tr": "ELRIS 2B"},
    "menu.file": {"en": "File", "tr": "Dosya"},
    "menu.edit": {"en": "Edit", "tr": "Düzen"},
    "menu.view": {"en": "View", "tr": "Görünüm"},
    "menu.export": {"en": "Export", "tr": "Dışa Aktar"},
    "menu.settings": {"en": "Settings", "tr": "Ayarlar"},
    "menu.language": {"en": "Language", "tr": "Dil"},
    "menu.help": {"en": "Help", "tr": "Yardım"},
    "action.open_folder": {"en": "Open folder…", "tr": "Klasör aç…"},
    "action.open_files": {"en": "Open files…", "tr": "Dosya aç…"},
    "action.save_result": {"en": "Save result", "tr": "Sonucu kaydet"},
    "action.export_image": {"en": "Export image…", "tr": "Görüntü dışa aktar…"},
    "action.quit": {"en": "Quit", "tr": "Çıkış"},
    "action.run": {"en": "Run inversion", "tr": "Ters çözümü başlat"},
    "action.run_selected": {"en": "Run selected", "tr": "Seçilenleri çöz"},
    "action.stop": {"en": "Stop", "tr": "Durdur"},
    "action.about": {"en": "About", "tr": "Hakkında"},
    "action.remove_from_list": {"en": "Remove from list", "tr": "Listeden kaldır"},
    "action.depth_map": {"en": "Build depth map", "tr": "Derinlik haritası oluştur"},
    "action.gen_loc": {"en": "Generate .loc files…", "tr": "Otomatik .loc oluştur…"},
    "lang.en": {"en": "English", "tr": "İngilizce"},
    "lang.tr": {"en": "Turkish", "tr": "Türkçe"},
    # --- centre tabs --------------------------------------------------
    "tab.line_view": {"en": "Line View", "tr": "Hat Görünümü"},
    "tab.data": {"en": "Data", "tr": "Veri"},
    "tab.map_view": {"en": "Map View", "tr": "Harita Görünümü"},
    "tab.depth_map": {"en": "Depth Map", "tr": "Derinlik Haritası"},
    "tab.view3d": {"en": "3D View", "tr": "3B Görünüm"},
    "panel.measured": {"en": "Measured apparent resistivity",
                       "tr": "Ölçülen görünür özdirenç"},
    "panel.calculated": {"en": "Calculated apparent resistivity",
                         "tr": "Hesaplanan görünür özdirenç"},
    "panel.model": {"en": "Inverted model", "tr": "Ters çözüm modeli"},
    "panel.no_selection": {"en": "Select a data file on the left.",
                           "tr": "Soldan bir veri dosyası seçin."},
    "panel.not_inverted": {"en": "Not inverted yet — run the inversion.",
                           "tr": "Henüz ters çözülmedi — ters çözümü başlatın."},
    # --- right tabs -------------------------------------------------
    "tab.visuals": {"en": "Visuals", "tr": "Görseller"},
    "tab.inversion": {"en": "Inversion", "tr": "Ters Çözüm"},
    "tab.processing": {"en": "Processing", "tr": "İşleme"},
    "tab.acquisition": {"en": "Acquisition", "tr": "Toplama"},
    "tab.contour_filters": {"en": "Contour & Filters", "tr": "Kontur ve Filtreler"},
    "vis.colormap": {"en": "Colormap", "tr": "Renk skalası"},
    "vis.load_colormap": {"en": "Load colour table (.clr)…",
                          "tr": "Renk skalası yükle (.clr)…"},
    "vis.reverse_colormap": {"en": "Reverse colormap", "tr": "Renk ölçeğini ters çevir"},
    "vis.data_range": {"en": "Data colour range (Ω·m)", "tr": "Veri renk aralığı (Ω·m)"},
    "vis.model_range": {"en": "Model colour range (Ω·m)", "tr": "Model renk aralığı (Ω·m)"},
    "vis.scale": {"en": "Scale", "tr": "Ölçek"},
    "vis.scale.log": {"en": "Logarithmic", "tr": "Logaritmik"},
    "vis.scale.linear": {"en": "Linear", "tr": "Doğrusal"},
    "vis.style": {"en": "Model style", "tr": "Model çizimi"},
    "vis.style.cells": {"en": "Voronoi cells", "tr": "Voronoi hücreleri"},
    "vis.style.contour": {"en": "Contour", "tr": "Kontur"},
    "vis.show_points": {"en": "Show data points", "tr": "Veri noktalarını göster"},
    "vis.show_electrodes": {"en": "Show electrodes", "tr": "Elektrotları göster"},
    "vis.show_contour_lines": {"en": "Contour lines", "tr": "Kontur çizgileri"},
    "vis.show_labels": {"en": "Contour labels", "tr": "Kontur etiketleri"},
    "vis.sensitivity": {"en": "Fade by sensitivity", "tr": "Duyarlılıkla soldur"},
    "vis.mask": {"en": "Blank unresolved corners", "tr": "Çözülmeyen köşeleri boşalt"},
    "vis.mask_pct": {"en": "Corner cut widening (%)", "tr": "Köşe kesimi genişletme (%)"},
    "vis.mask_shape": {"en": "Corner cut shape", "tr": "Köşe kesim şekli"},
    "vis.mask_shape.triangle": {"en": "Triangle", "tr": "Üçgen"},
    "vis.mask_shape.ellipse": {"en": "Ellipse", "tr": "Elips"},
    "inv.discretization": {"en": "Model discretization", "tr": "Model ayrıklaştırma"},
    "inv.cell_type": {"en": "Cell type", "tr": "Hücre tipi"},
    "inv.cell.voronoi": {"en": "Voronoi polygons", "tr": "Voronoi poligonları"},
    "inv.cell.triangle": {"en": "Triangles", "tr": "Üçgenler"},
    "inv.cell.rect": {"en": "Rectangular blocks", "tr": "Dikdörtgen bloklar"},
    "inv.mesh_wing": {"en": "Side extension (× spacing)", "tr": "Yan uzatma (× aralık)"},
    "inv.mesh_depth": {"en": "Depth factor (× z_max)", "tr": "Derinlik katsayısı (× z_max)"},
    "inv.surface_div": {"en": "Surface refinement", "tr": "Yüzey sıklığı"},
    "inv.method": {"en": "Inversion method", "tr": "Ters çözüm yöntemi"},
    "inv.lambda_mode": {"en": "λ strategy", "tr": "λ stratejisi"},
    "inv.mode.search": {"en": "Adaptive (line search)", "tr": "Uyarlanır (hat arama)"},
    "inv.mode.matlab": {"en": "Fixed + cooling (MATLAB)", "tr": "Sabit + soğutma (MATLAB)"},
    "inv.lambda0": {"en": "Initial λ (auto if blank)", "tr": "Başlangıç λ (boşsa oto)"},
    "inv.cooling": {"en": "λ cooling factor", "tr": "λ soğutma katsayısı"},
    "inv.target_rms": {"en": "Target RMS % (blank = auto)",
                       "tr": "Hedef RMS % (boş = oto)"},
    "inv.progress": {"en": "Progress", "tr": "İlerleme"},
    "inv.max_iter": {"en": "Max iterations", "tr": "En fazla yineleme"},
    "inv.tol": {"en": "Stop at ΔRMS <", "tr": "Durdurma: ΔRMS <"},
    "acq.array": {"en": "Array type", "tr": "Elektrot dizilimi"},
    "acq.spacing": {"en": "Electrode spacing", "tr": "Elektrot aralığı"},
    "acq.n_electrodes": {"en": "Electrodes", "tr": "Elektrot sayısı"},
    "acq.n_data": {"en": "Data points", "tr": "Veri sayısı"},
    "acq.rhoa_range": {"en": "ρₐ range", "tr": "ρₐ aralığı"},
    "acq.pseudo_depth": {"en": "Max pseudo-depth", "tr": "En büyük yapma-derinlik"},
    "proc.none": {"en": "No processing steps recorded.",
                  "tr": "Kayıtlı işleme adımı yok."},
    "axis.distance": {"en": "Distance (m)", "tr": "Uzaklık (m)"},
    "axis.pseudo_depth": {"en": "Pseudo-depth (m)", "tr": "Yapma-derinlik (m)"},
    "axis.depth": {"en": "Depth (m)", "tr": "Derinlik (m)"},
    "axis.iteration": {"en": "iteration", "tr": "yineleme"},
    "axis.rms": {"en": "RMS misfit (%)", "tr": "RMS uyumsuzluk (%)"},
    "axis.count": {"en": "count", "tr": "sayı"},
    "axis.rhoa_meas": {"en": r"measured $\rho_a$ ($\Omega\cdot$m)",
                       "tr": r"ölçülen $\rho_a$ ($\Omega\cdot$m)"},
    "axis.rhoa_calc": {"en": r"calculated $\rho_a$ ($\Omega\cdot$m)",
                       "tr": r"hesaplanan $\rho_a$ ($\Omega\cdot$m)"},
    "axis.easting": {"en": "Easting (m)", "tr": "Doğu (m)"},
    "axis.northing": {"en": "Northing (m)", "tr": "Kuzey (m)"},
    "axis.x_dist": {"en": "X distance (m)", "tr": "X yönü uzaklık (m)"},
    "axis.y_dist": {"en": "Y distance (m)", "tr": "Y yönü uzaklık (m)"},
    "cbar.rhoa": {"en": r"$\rho_a$ ($\Omega\cdot$m)", "tr": r"$\rho_a$ ($\Omega\cdot$m)"},
    "cbar.rho": {"en": r"$\rho$ ($\Omega\cdot$m)", "tr": r"$\rho$ ($\Omega\cdot$m)"},
    "inv.robust": {"en": "Robust data weighting", "tr": "Dayanıklı veri ağırlıklandırma"},
    "inv.robust_min": {"en": "Min. data weight", "tr": "En düşük veri ağırlığı"},
    "inv.robust.none": {"en": "Off", "tr": "Kapalı"},
    "inv.robust.huber": {"en": "Huber", "tr": "Huber"},
    "inv.robust.cauchy": {"en": "Cauchy", "tr": "Cauchy"},
    "inv.robust.bisquare": {"en": "Bisquare", "tr": "Bisquare"},
    "map.no_loc": {"en": "no .loc files", "tr": ".loc dosyası yok"},
    "map.start": {"en": "Line start", "tr": "Profil başı"},
    "map.end": {"en": "Line end", "tr": "Profil sonu"},
    "panel.no_depth_map": {
        "en": "Select lines with .loc files (Map View) and click "
              "“Build depth map”.",
        "tr": ".loc dosyası olan hatları seçip “Derinlik haritası "
              "oluştur”'a tıklayın."},
    "depthmap.missing_loc": {
        "en": "Missing .loc file for: {names}",
        "tr": "Şu dosyalarda .loc dosyası yok: {names}"},
    "depthmap.need_two": {
        "en": "Need at least two successfully inverted lines with .loc files.",
        "tr": "En az iki başarıyla ters çözülmüş, .loc'lu hat gerekiyor."},
    "depthmap.done": {"en": "Depth map built from {n} line(s)",
                      "tr": "{n} hattan derinlik haritası oluşturuldu"},
    "depthmap.title": {"en": "Depth {depth:.2f} m", "tr": "Derinlik {depth:.2f} m"},
    "depthmap.depth": {"en": "Depth", "tr": "Derinlik"},
    "depthmap.prev": {"en": "Previous depth", "tr": "Önceki derinlik"},
    "depthmap.next": {"en": "Next depth", "tr": "Sonraki derinlik"},
    "depthmap.no_data": {"en": "No data at this depth", "tr": "Bu derinlikte veri yok"},
    "depthmap.anim_step": {"en": "Step", "tr": "Adım"},
    "depthmap.build_animation": {"en": "Build animation…", "tr": "Animasyon oluştur…"},
    "depthmap.anim_save": {"en": "Save animation", "tr": "Animasyonu kaydet"},
    "depthmap.anim_none": {"en": "Build a depth map first.",
                           "tr": "Önce bir derinlik haritası oluşturun."},
    "depthmap.anim_too_few": {
        "en": "Step is larger than the available depth range.",
        "tr": "Adım, mevcut derinlik aralığından büyük."},
    "depthmap.anim_done": {
        "en": "Animation saved: {n} frame(s) + {n} PNG slice(s) (ertslice_*.png)",
        "tr": "Animasyon kaydedildi: {n} kare + {n} PNG dilim (ertslice_*.png)"},
    "depthmap.show_lines": {"en": "Show lines", "tr": "Hat çizgilerini göster"},
    "depthmap.show_names": {"en": "Show line names", "tr": "Hat isimlerini göster"},
    # --- contour & filters -----------------------------------------
    "contour.group": {"en": "Contouring", "tr": "Konturlama"},
    "contour.interp": {"en": "Interpolation", "tr": "Enterpolasyon"},
    "contour.interp.triangulation": {"en": "Triangulation (no extrapolation)",
                                     "tr": "Üçgenleme (dış bölgeye taşırmaz)"},
    "contour.interp.linear": {"en": "Grid – linear", "tr": "Izgara – doğrusal"},
    "contour.interp.cubic": {"en": "Grid – cubic", "tr": "Izgara – kübik"},
    "contour.interp.nearest": {"en": "Grid – nearest", "tr": "Izgara – en yakın"},
    "contour.n_levels": {"en": "Contour levels", "tr": "Kontur seviyesi"},
    "contour.fill": {"en": "Filled contours", "tr": "Dolgulu kontur"},
    "contour.grid_step": {"en": "Grid cell size", "tr": "Izgara hücre boyutu"},
    "contour.filters_group": {"en": "Filters", "tr": "Filtreler"},
    "contour.smooth": {"en": "Smoothing", "tr": "Yumuşatma"},
    "contour.smooth_sigma": {"en": "Smoothing strength", "tr": "Yumuşatma şiddeti"},
    "contour.sharpen": {"en": "Sharpening", "tr": "Keskinleştirme"},
    "contour.sharpen_amount": {"en": "Sharpening amount", "tr": "Keskinleştirme miktarı"},
    "contour.emboss": {"en": "Emboss", "tr": "Kabartma"},
    "contour.emboss_strength": {"en": "Emboss strength", "tr": "Kabartma şiddeti"},
    "contour.hillshade": {"en": "Hillshade", "tr": "Gölgelendirme"},
    "contour.hs_azimuth": {"en": "Light azimuth", "tr": "Işık azimutu"},
    "contour.hs_altitude": {"en": "Light altitude", "tr": "Işık yükseltisi"},
    "contour.hs_strength": {"en": "Shading strength", "tr": "Gölgelendirme şiddeti"},
    # --- 3D view -----------------------------------------------------
    "view3d.iso_values": {"en": "Isosurface values (Ω·m)",
                          "tr": "Eş-yüzey değerleri (Ω·m)"},
    "view3d.iso_placeholder": {"en": "e.g. 50, 150, 300 (blank = auto)",
                               "tr": "örn. 50, 150, 300 (boş = otomatik)"},
    "view3d.show_iso": {"en": "Isosurfaces", "tr": "Eş-yüzeyler"},
    "view3d.show_outline": {"en": "Bounding box", "tr": "Sınır kutusu"},
    "view3d.show_grid": {"en": "Axis grid + labels", "tr": "Eksen ızgarası + etiketler"},
    "view3d.grid_step": {"en": "Grid spacing", "tr": "Izgara aralığı"},
    "view3d.axis_x_title": {"en": "X (m)", "tr": "X (m)"},
    "view3d.axis_y_title": {"en": "Y (m)", "tr": "Y (m)"},
    "view3d.axis_z_title": {"en": "Depth (m, − down)", "tr": "Derinlik (m, − aşağı)"},
    "view3d.axis_z": {"en": "Z (depth)", "tr": "Z (derinlik)"},
    "view3d.slice_count": {"en": "count", "tr": "adet"},
    "view3d.depth_fmt": {"en": "Depth = {:.2f} m", "tr": "Derinlik = {:.2f} m"},
    "data.misfit_section": {"en": "Per-datum misfit (%)",
                            "tr": "Veri noktası uyumsuzluğu (%)"},
    "data.convergence": {"en": "Convergence", "tr": "Yakınsama"},
    "data.residuals": {"en": "Residuals", "tr": "Artıklar"},
    "data.crossplot": {"en": "Measured vs calculated", "tr": "Ölçülen–hesaplanan"},
    "data.rhoa_hist": {"en": "ρₐ distribution", "tr": "ρₐ dağılımı"},
    # --- status ---------------------------------------------------
    "status.ready": {"en": "Ready", "tr": "Hazır"},
    "status.loaded": {"en": "Loaded {name}", "tr": "{name} yüklendi"},
    "status.running": {"en": "Inverting {name} …", "tr": "{name} ters çözülüyor …"},
    "status.iter": {"en": "{name}  iter {it}  RMS {rms:.2f}%",
                    "tr": "{name}  yineleme {it}  RMS %{rms:.2f}"},
    "status.done": {"en": "{name}  done — RMS {rms:.2f}%",
                    "tr": "{name}  bitti — RMS %{rms:.2f}"},
    "status.batch_done": {"en": "Finished {n} file(s)", "tr": "{n} dosya bitti"},
    "status.error": {"en": "Error: {msg}", "tr": "Hata: {msg}"},
    "tree.header": {"en": "Project / files", "tr": "Proje / dosyalar"},
    # --- .loc generator ------------------------------------------------
    "locgen.title": {"en": "Generate .loc files", "tr": "Otomatik .loc oluştur"},
    "locgen.origin_x": {"en": "Origin X (first profile, electrode 1)",
                        "tr": "Orijin X (ilk profil, 1. elektrot)"},
    "locgen.origin_y": {"en": "Origin Y (first profile, electrode 1)",
                        "tr": "Orijin Y (ilk profil, 1. elektrot)"},
    "locgen.azimuth": {"en": "Azimuth (0°=N, 90°=E, clockwise)",
                       "tr": "Azimut (0°=K, 90°=D, saat yönü)"},
    "locgen.line_spacing": {"en": "Line spacing", "tr": "Hat aralığı"},
    "locgen.side": {"en": "Next lines shift to the", "tr": "Sonraki hatlar hangi tarafa kaysın"},
    "locgen.side.right": {"en": "Right of travel direction",
                          "tr": "Gidiş yönünün sağına"},
    "locgen.side.left": {"en": "Left of travel direction",
                         "tr": "Gidiş yönünün soluna"},
    "locgen.info": {"en": "{n} profile(s), in order: {names}. "
                         "The first one starts exactly at the origin.",
                    "tr": "{n} profil, sırayla: {names}. "
                         "İlki tam orijinden başlar."},
    "locgen.skipped": {"en": " ({n} skipped - not loaded)",
                       "tr": " ({n} tanesi atlanacak - yüklenmemiş)"},
    "locgen.none_selected": {"en": "Select one or more .dat files first (tree, left).",
                            "tr": "Önce soldaki ağaçtan bir veya daha fazla .dat dosyası seçin."},
    "locgen.done": {"en": "{n} .loc file(s) written", "tr": "{n} .loc dosyası yazıldı"},
}


class Translator(QObject):
    language_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self._lang = "tr"

    @property
    def language(self) -> str:
        return self._lang

    def set_language(self, lang: str) -> None:
        if lang not in ("en", "tr") or lang == self._lang:
            return
        self._lang = lang
        self.language_changed.emit(lang)

    def t(self, key: str, **fmt) -> str:
        entry = _STR.get(key)
        text = entry.get(self._lang, entry.get("en")) if entry else key
        return text.format(**fmt) if fmt else text


translator = Translator()


def t(key: str, **fmt) -> str:
    return translator.t(key, **fmt)
