import streamlit as st
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
from shapely.geometry import mapping
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as mticker
from pyproj import Transformer
import tempfile
import os
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import contextily as cx
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D # Import Line2D for adding lines to axes

# --- Streamlit App Setup ---
st.set_page_config(layout="wide", page_title="Peta Rawan Banjir Kota Malang")

st.title("🗺️ Peta Rawan Banjir Kota Malang")
st.write("Aplikasi ini memvisualisasikan tingkat kerawanan banjir di Kota Malang berdasarkan beberapa faktor geografis dan lingkungan.")

# --- File Paths (Adjust these paths based on your deployment environment) ---
shapefile_path = "data/Malang.shp"
river_shapefile_path = "data/SUNGAI_AR_25K.shp"

raster_files = {
    "curah_hujan": "data/Skoringg_CHH (1).tif",
    "ketinggian": "data/Skoring_DEMM.tif",
    "kemiringan": "data/Skoring_Slop.tif",
    "tutupan_lahan": "data/Skoring_DynamicWorld.tif",
    "jarak_sungai": "data/buffer_sungai_100m (1).tif"
}

# --- Fixed Weights (Not adjustable by user) ---
fixed_weights = {
  "curah_hujan": 0.25,
    "jarak_sungai": 0.15,
    "kemiringan": 0.20,
    "ketinggian": 0.20,
    "tutupan_lahan": 0.20
    
}

# Ensure weights sum to 1 (normalization for fixed weights)
total_fixed_weight = sum(fixed_weights.values())
if total_fixed_weight > 0:
    normalized_fixed_weights = {k: v / total_fixed_weight for k, v in fixed_weights.items()}
else:
    normalized_fixed_weights = {k: 0 for k in fixed_weights.keys()}
    st.error("Total bobot yang ditetapkan adalah 0. Pastikan bobot memiliki nilai positif.")


# --- Data Loading and Processing (Cached for performance) ---
@st.cache_data
def load_and_process_data(shapefile_p, river_shapefile_p, raster_fs, wts):
    # Load shapefile administrasi Malang
    try:
        gdf = gpd.read_file(shapefile_p)
        gdf_malang = gdf
        if gdf_malang.empty:
            st.error("Wilayah 'Malang' tidak ditemukan! Harap periksa path shapefile.")
            return None, None, None, None, None, None
    except Exception as e:
        st.error(f"Error memuat shapefile administrasi Malang: {e}")
        return None, None, None, None, None, None

    # Load shapefile jalur sungai
    gdf_rivers = None
    try:
        gdf_rivers = gpd.read_file(river_shapefile_p)
        if gdf_rivers.crs != gdf_malang.crs:
            gdf_rivers = gdf_rivers.to_crs(gdf_malang.crs)
        # It's better to clip first, then reproject for length calculations later
        gdf_rivers_clipped = gpd.clip(gdf_rivers, gdf_malang)
        if len(gdf_rivers_clipped) == 0:
            st.warning("PERINGATAN: Tidak ada sungai yang berada dalam area Malang. Peta akan dibuat tanpa jalur sungai.")
            gdf_rivers = None
        else:
            gdf_rivers = gdf_rivers_clipped
    except Exception as e:
        st.error(f"Error memuat shapefile sungai: {e}. Peta akan dibuat tanpa jalur sungai.")
        gdf_rivers = None

    # Referensi raster
    ref_path = list(raster_fs.values())[0]
    ref_crs = None
    ref_transform = None
    ref_shape = None
    ref_meta = None
    pixel_size = None

    try:
        with rasterio.open(ref_path) as ref:
            ref_crs = ref.crs
            ref_transform = ref.transform
            ref_shape = (ref.height, ref.width)
            ref_meta = ref.meta.copy()
            if ref_crs and ref_crs.is_projected:
                pixel_size = abs(ref_transform.a * ref_transform.e)
    except Exception as e:
        st.error(f"Error memuat raster referensi ({ref_path}): {e}")
        return None, None, None, None, None, None

    skor_total = np.zeros((ref_shape[0], ref_shape[1]), dtype='float32')
    out_meta = ref_meta.copy()
    out_meta.update({"dtype": "float32", "count": 1, "nodata": np.nan})


    progress_bar = st.progress(0)
    for i, (name, path) in enumerate(raster_fs.items()):
        weight = wts.get(name, 0)
        if weight == 0:
            st.warning(f"Bobot untuk '{name}' adalah 0. Lapisan ini tidak akan berkontribusi pada skor.")
            progress_bar.progress((i + 1) / len(raster_files))
            continue

        try:
            with rasterio.open(path) as src:
                gdf_masked = gdf_malang.to_crs(src.crs) if gdf_malang.crs != src.crs else gdf_malang
                shapes = [mapping(geom) for geom in gdf_masked.geometry]
                masked, _ = mask(src, shapes, crop=True)
                masked = masked.astype('float32')
                if src.nodata is not None:
                    masked[masked == src.nodata] = np.nan

                resampled = np.empty((ref_shape[0], ref_shape[1]), dtype='float32')
                reproject(
                    source=masked,
                    destination=resampled,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear
                )

                valid_mask = ~np.isnan(resampled)
                skor_total[valid_mask] += (resampled[valid_mask] * weight)
        except Exception as e:
            st.warning(f"Error processing {name}: {e}. Skipping this layer.")
        progress_bar.progress((i + 1) / len(raster_files))

    if gdf_malang.crs != ref_crs:
        gdf_malang = gdf_malang.to_crs(ref_crs)
    shapes = [mapping(geom) for geom in gdf_malang.geometry]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = os.path.join(tmpdir, "temp_skor.tif")
        with rasterio.open(tmpfile, "w", **out_meta) as dst:
            dst.write(skor_total, 1)

        with rasterio.open(tmpfile) as src_tmp:
            masked_data, _, = mask(src_tmp, shapes, crop=False, nodata=np.nan)
    skor_total_masked = masked_data[0]

    # Klasifikasi Rawan Banjir
    klasifikasi = np.full(skor_total_masked.shape, np.nan)
    valid_scores = skor_total_masked[~np.isnan(skor_total_masked)]

    if len(valid_scores) > 0:
        # Menggunakan kuartil untuk klasifikasi yang lebih robust
        q25 = np.percentile(valid_scores, 25)
        q50 = np.percentile(valid_scores, 50)
        q75 = np.percentile(valid_scores, 75)

        # Mengubah batas klasifikasi berdasarkan kuartil
        klasifikasi[~np.isnan(skor_total_masked) & (skor_total_masked <= q25)] = 1  # Rawan Rendah
        klasifikasi[~np.isnan(skor_total_masked) & (skor_total_masked > q25) & (skor_total_masked <= q75)] = 2  # Rawan Sedang
        klasifikasi[~np.isnan(skor_total_masked) & (skor_total_masked > q75)] = 3  # Rawan Tinggi
    else:
        st.warning("Tidak ada data skor yang valid untuk diklasifikasikan.")

    return klasifikasi, gdf_malang, gdf_rivers, ref_transform, ref_crs, pixel_size

# --- Sidebar for User Input (Only visibility and basemap) ---
st.sidebar.header("Pengaturan Tampilan Peta")

st.sidebar.subheader("Pilihan Tampilan")
show_rivers = st.sidebar.checkbox("Tampilkan Jalur Sungai", value=True)

# Corrected basemap providers (Stamen.Terrain removed)
selected_basemap = st.sidebar.selectbox(
    "Pilih Basemap",
    options=[
        "OpenStreetMap.Mapnik",
        "Esri.WorldImagery",
        "CartoDB.Positron"
    ],
    index=0
)

# Load and process data with fixed weights
klasifikasi, gdf_malang, gdf_rivers, ref_transform, ref_crs, pixel_size = load_and_process_data(
    shapefile_path, river_shapefile_path, raster_files, normalized_fixed_weights
)

if klasifikasi is not None and gdf_malang is not None:
    # Koordinat untuk plotting
    rows, cols = klasifikasi.shape
    plot_extent = (ref_transform.c, ref_transform.c + ref_transform.a * cols,
                   ref_transform.f + ref_transform.e * rows, ref_transform.f)

    plot_crs = ref_crs
    if plot_crs and plot_crs.to_epsg() != 4326:
        transformer = Transformer.from_crs(plot_crs, "EPSG:4326", always_xy=True)
        xmin_lon, ymin_lat = transformer.transform(plot_extent[0], plot_extent[2])
        xmax_lon, ymax_lat = transformer.transform(plot_extent[1], plot_extent[3])
        plot_extent_wgs84 = (xmin_lon, xmax_lon, ymin_lat, ymax_lat)
    else:
        plot_extent_wgs84 = plot_extent

    # --- Setup Layout Kolom Streamlit ---
    col_description, col_map = st.columns([1, 2])

    with col_description:
        st.subheader("Deskripsi dan Informasi Peta")
        st.write("""
        Peta ini menampilkan tingkat kerawanan banjir di Kota Malang yang dihitung berdasarkan
        berbagai faktor geografis dan lingkungan. Klasifikasi dibagi menjadi tiga kategori:
        """)
        st.markdown("""
        * <span style="color:#2ECC71; font-weight:bold;">Rawan Rendah</span>: Area dengan risiko banjir paling rendah, cenderung aman dari genangan signifikan.
        * <span style="color:#F39C12; font-weight:bold;">Rawan Sedang</span>: Area dengan risiko banjir menengah, perlu kewaspadaan terutama saat curah hujan tinggi.
        * <span style="color:#E74C3C; font-weight:bold;">Rawan Tinggi</span>: Area dengan risiko banjir tinggi, sangat rentan terhadap genangan dan memerlukan perhatian serta tindakan mitigasi khusus.
        """, unsafe_allow_html=True)
        st.write("---")
        st.write("""
        **Metodologi Analisis Kerawanan Banjir:**
        Analisis kerawanan banjir ini dilakukan menggunakan pendekatan **Sistem Informasi Geografis (SIG)**
        dengan metode **pembobotan (weighted overlay)** dari beberapa parameter kunci yang memengaruhi
        potensi terjadinya banjir. Setiap parameter memiliki bobot yang berbeda berdasarkan
        tingkat kontribusinya terhadap risiko banjir.
        """)
        st.write("""
        **Parameter Utama dan Bobotnya:**
        Berikut adalah faktor-faktor geografis dan lingkungan yang digunakan dalam analisis ini,
        disertai dengan bobot kontribusi masing-masing:
        """)
        for factor, weight in normalized_fixed_weights.items():
            st.markdown(f"- **{factor.replace('_', ' ').title()}**: Berkontribusi **{weight*100:.1f}%** terhadap skor kerawanan banjir.")

        st.write("""
        Setiap parameter telah dinormalisasi dan diberikan skor berdasarkan tingkat pengaruhnya terhadap banjir.
        Kemudian, skor-skor ini digabungkan menggunakan bobot yang telah ditentukan untuk menghasilkan
        indeks kerawanan banjir. Indeks ini selanjutnya diklasifikasikan ke dalam tiga kategori
        untuk kemudahan interpretasi visual.
        """)
        st.info("""
        Peta ini bertujuan untuk memberikan gambaran awal potensi banjir dan sebaiknya digunakan sebagai panduan
        untuk perencanaan wilayah dan mitigasi bencana. **Data yang digunakan berasal dari berbagai
        sumber geografis dan mungkin memerlukan pembaruan berkala** untuk akurasi yang lebih tinggi.
        """)

    with col_map:
        st.subheader("Visualisasi Peta Rawan Banjir")

        # --- GridSpec for the new layout ---
        # Back to 2 rows, 2 columns, but with adjusted height_ratios
        fig = plt.figure(figsize=(25, 20)) # Significantly increased figure size
        gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 0.25], height_ratios=[1, 0.05], wspace=0.1, hspace=0.0) # Adjusted ratios, hspace=0 to remove space between rows

        # Main map axis (row 0, column 0)
        ax_map = fig.add_subplot(gs[0, 0])

        # Single Legend/Info axis (row 0, column 1)
        ax_legend_and_info = fig.add_subplot(gs[0, 1])
        ax_legend_and_info.set_axis_off()

        # Removed ax_bottom_info_weights and ax_info as they are now consolidated


        # Definisi colormap dan normalisasi
        cmap = colors.ListedColormap(['#2ECC71', '#F39C12', '#E74C3C'])
        bounds = [0.5, 1.5, 2.5, 3.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        # Plot klasifikasi on ax_map
        im = ax_map.imshow(klasifikasi, cmap=cmap, norm=norm,
                           extent=plot_extent_wgs84,
                           origin='upper', interpolation='none', alpha=0.8)

        # Tambahkan basemap (dynamically selected) on ax_map
        try:
            basemap_provider = getattr(cx.providers, selected_basemap.split('.')[0])
            if len(selected_basemap.split('.')) > 1:
                basemap_provider = getattr(basemap_provider, selected_basemap.split('.')[1])
            cx.add_basemap(ax_map, crs=gdf_malang.to_crs("EPSG:4326").crs, source=basemap_provider)
        except Exception as e:
            st.warning(f"Tidak dapat memuat basemap '{selected_basemap}': {e}. Menggunakan OpenStreetMap.Mapnik sebagai default.")
            cx.add_basemap(ax_map, crs=gdf_malang.to_crs("EPSG:4326").crs, source=cx.providers.OpenStreetMap.Mapnik)

        # Plot batas kota on ax_map
        gdf_plot = gdf_malang.to_crs("EPSG:4326")
        shp_boundary_plot = gdf_plot.boundary.plot(ax=ax_map, edgecolor='black', linewidth=2.5) # Increased linewidth

        # Plot jalur sungai (conditional based on sidebar checkbox) on ax_map
        if show_rivers and gdf_rivers is not None:
            try:
                gdf_rivers_plot = gdf_rivers.to_crs("EPSG:4326")
                if len(gdf_rivers_plot) > 0:
                    gdf_rivers_plot.plot(ax=ax_map, color='white', linewidth=5, alpha=0.7, zorder=3) # Increased linewidth
                    gdf_rivers_plot.plot(ax=ax_map, color='#1E88E5', linewidth=3, # Increased linewidth
                                         alpha=0.9, linestyle='-', zorder=4)

                    nama_kolom_sungai = None
                    possible_name_columns = ['NAMA', 'NAME', 'REMARK', 'NAMOBJ', 'RIVER_NAME', 'NAMA_SUNGAI']
                    for col in possible_name_columns:
                        if col in gdf_rivers_plot.columns:
                            nama_kolom_sungai = col
                            break

                    if nama_kolom_sungai and not gdf_rivers_plot.empty:
                        # Reproject for accurate length calculation before filtering for labels
                        # Menggunakan CRS yang lebih sesuai untuk Indonesia, misal UTM Zone 49S (EPSG:32749)
                        gdf_rivers_for_label_filter = gdf_rivers_plot.to_crs("EPSG:32749")
                        # Filter sungai yang lebih panjang dari 1 km untuk label
                        rivers_to_label = gdf_rivers_for_label_filter[gdf_rivers_for_label_filter.geometry.length > 1000]
                        # Reproject back to 4326 for plotting centroids
                        rivers_to_label_wgs84 = rivers_to_label.to_crs("EPSG:4326")

                        for idx, row in rivers_to_label_wgs84.iterrows():
                            if pd.notna(row[nama_kolom_sungai]) and hasattr(row.geometry, 'centroid'):
                                try:
                                    centroid = row.geometry.centroid
                                    if (plot_extent_wgs84[0] <= centroid.x <= plot_extent_wgs84[1] and
                                        plot_extent_wgs84[2] <= centroid.y <= plot_extent_wgs84[3]):
                                        ax_map.text(centroid.x, centroid.y, str(row[nama_kolom_sungai]),
                                                    fontsize=8, ha='center', va='center', # Increased fontsize
                                                    color='#0D47A1', weight='bold', zorder=6,
                                                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
                                except Exception as label_error:
                                    pass
            except Exception as e:
                st.error(f"Error menampilkan sungai: {e}")

        # Set zorder for plot elements
        im.set_zorder(2)
        for collection in shp_boundary_plot.collections:
            collection.set_zorder(5)

        # Add region labels (if 'NAMOBJ' column exists)
        if 'NAMOBJ' in gdf_plot.columns:
            for idx, row in gdf_plot.iterrows():
                if hasattr(row.geometry, 'centroid'):
                    centroid = row.geometry.centroid
                    if (plot_extent_wgs84[0] <= centroid.x <= plot_extent_wgs84[1] and
                        plot_extent_wgs84[2] <= centroid.y <= plot_extent_wgs84[3]):
                        ax_map.text(centroid.x, centroid.y, row["NAMOBJ"],
                                    fontsize=9, ha='center', va='center', # Increased fontsize
                                    color='black', weight='bold', zorder=6)

        # Title and labels for ax_map
        ax_map.set_title("PETA RAWAN BANJIR KOTA MALANG, PROVINSI JAWA TIMUR",
                        fontsize=18, weight='bold', pad=15) # Increased fontsize and pad
        ax_map.set_xlabel("Longitude", fontsize=12) # Increased fontsize
        ax_map.set_ylabel("Latitude", fontsize=12) # Increased fontsize

        # Grid for ax_map
        ax_map.grid(True, linestyle='--', alpha=0.3, color='gray')

        # Format axis for ax_map
        ax_map.xaxis.set_major_formatter(mticker.FuncFormatter(lambda val, pos: f'{val:.3f}°'))
        ax_map.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, pos: f'{val:.3f}°'))

        # Scalebar on ax_map
        try:
            fontprops = fm.FontProperties(size=10) # Increased font size for scalebar
            center_lon = (plot_extent_wgs84[0] + plot_extent_wgs84[1]) / 2
            center_lat = (plot_extent_wgs84[2] + plot_extent_wgs84[3]) / 2
            scale_length_degrees = 5000 / (111320 * np.cos(np.deg2rad(center_lat)))
            scalebar = AnchoredSizeBar(ax_map.transData, # Attach to ax_map
                                        scale_length_degrees,
                                        '5 km',
                                        'lower left',
                                        pad=0.5,
                                        color='black',
                                        frameon=True,
                                        size_vertical=0.002,
                                        fontproperties=fontprops,
                                        zorder=7)
            ax_map.add_artist(scalebar)
        except Exception as e:
            st.warning(f"Scalebar tidak dapat ditambahkan: {e}")

        # North Arrow on ax_map
        try:
            img_path = "data/Arah Mata Angin.jpeg"
            if os.path.exists(img_path):
                img = mpimg.imread(img_path)
                img_x = plot_extent_wgs84[0] + 0.9 * (plot_extent_wgs84[1] - plot_extent_wgs84[0])
                img_y = plot_extent_wgs84[3] - 0.1 * (plot_extent_wgs84[3] - plot_extent_wgs84[2])
                imagebox = OffsetImage(img, zoom=0.08) # Slightly increased zoom
                ab = AnnotationBbox(imagebox, (img_x, img_y), frameon=False, zorder=7,
                                     xycoords='data', boxcoords="data", pad=0)
                ax_map.add_artist(ab)
            else:
                st.warning(f"Gambar kompas tidak ditemukan di {img_path}. Melewatkan kompas.")
        except Exception as e:
            st.warning(f"Kompas tidak dapat ditambahkan: {e}")

        ax_map.set_aspect('equal')

        # --- ALL Legend and Info in ax_legend_and_info ---

        # Kelas Rawan Banjir Legend (top of ax_legend_and_info)
        ax_legend_and_info.text(0.0, 0.98, "Kelas Rawan Banjir:", fontsize=13, weight='bold', transform=ax_legend_and_info.transAxes)

        # Manually plot rectangles for custom colorbar
        y_start_color = 0.98 # Starting Y for labels
        y_spacing_color = 0.07 # Adjusted spacing
        box_width = 0.15 # Adjusted width
        box_height = 0.04 # Adjusted height
        label_x_offset = box_width + 0.02 # Offset for labels from boxes

        # Rawan Rendah
        ax_legend_and_info.add_patch(plt.Rectangle((0.0, y_start_color - y_spacing_color), box_width, box_height,
                                             facecolor='#2ECC71', edgecolor='black', transform=ax_legend_and_info.transAxes))
        ax_legend_and_info.text(label_x_offset, y_start_color - y_spacing_color + box_height/2, 'Rawan Rendah',
                               va='center', fontsize=11, transform=ax_legend_and_info.transAxes)

        # Rawan Sedang
        ax_legend_and_info.add_patch(plt.Rectangle((0.0, y_start_color - 2*y_spacing_color), box_width, box_height,
                                             facecolor='#F39C12', edgecolor='black', transform=ax_legend_and_info.transAxes))
        ax_legend_and_info.text(label_x_offset, y_start_color - 2*y_spacing_color + box_height/2, 'Rawan Sedang',
                               va='center', fontsize=11, transform=ax_legend_and_info.transAxes)

        # Rawan Tinggi
        ax_legend_and_info.add_patch(plt.Rectangle((0.0, y_start_color - 3*y_spacing_color), box_width, box_height,
                                             facecolor='#E74C3C', edgecolor='black', transform=ax_legend_and_info.transAxes))
        ax_legend_and_info.text(label_x_offset, y_start_color - 3*y_spacing_color + box_height/2, 'Rawan Tinggi',
                               va='center', fontsize=11, transform=ax_legend_and_info.transAxes)

        # --- Infrastruktur: Informasi (below Rawan Banjir Legend) ---
        infra_y_start = y_start_color - 3*y_spacing_color - 0.1 # Start below last color legend item
        ax_legend_and_info.text(0.0, infra_y_start, "Infrastruktur:", fontsize=13, weight='bold', transform=ax_legend_and_info.transAxes)

        # Jalur Sungai legend
        river_y_pos_infra = infra_y_start - 0.07 # Adjusted spacing
        ax_legend_and_info.add_line(Line2D([0, 0.03], [river_y_pos_infra, river_y_pos_infra], # Shorter line
                                           color='#1E88E5', linewidth=3, transform=ax_legend_and_info.transAxes))
        ax_legend_and_info.text(0.04, river_y_pos_infra, "Jalur Sungai", va='center', fontsize=11, transform=ax_legend_and_info.transAxes)

        # Batas Kota Malang legend
        boundary_y_pos_infra = river_y_pos_infra - 0.07 # Below river, adjusted spacing
        ax_legend_and_info.add_line(Line2D([0, 0.03], [boundary_y_pos_infra, boundary_y_pos_infra], # Shorter line
                                           color='black', linewidth=2.5, transform=ax_legend_and_info.transAxes))
        ax_legend_and_info.text(0.04, boundary_y_pos_infra, "Batas Kota Malang", va='center', fontsize=11, transform=ax_legend_and_info.transAxes)


        # --- Bobot Analisis (below Infrastruktur) ---
        weights_y_start = boundary_y_pos_infra - 0.1 # Start below last infra item
        ax_legend_and_info.text(0.0, weights_y_start, "Bobot Analisis:", fontsize=13, weight='bold', transform=ax_legend_and_info.transAxes)
        weight_item_y_start = weights_y_start - 0.07 # Adjusted spacing for first item
        for factor, weight in normalized_fixed_weights.items():
            ax_legend_and_info.text(0.0, weight_item_y_start, # Aligned left
                                 f"* {factor.replace('_', ' ').title()}: {weight*100:.1f}%",
                                 fontsize=11, transform=ax_legend_and_info.transAxes)
            weight_item_y_start -= 0.07 # Adjusted spacing for subsequent items


        plt.tight_layout(rect=[0, 0, 1, 0.98])
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Statistik Area Rawan Banjir")

    if pixel_size is not None and klasifikasi is not None:
        try:
            area_unit = "m²"
            pixel_area_sq_unit = pixel_size
            if ref_crs.is_projected:
                if 'metre' in str(ref_crs.coordinate_system.axis_info[0].unit_name).lower():
                    pixel_area_sq_unit = pixel_size / 1_000_000 # Convert m² to km²
                    area_unit = "km²"
                else:
                    area_unit = f"({ref_crs.coordinate_system.axis_info[0].unit_name})²"
            else:
                st.warning("CRS referensi adalah geografis (derajat). Perhitungan area mungkin tidak akurat.")
                area_unit = "derajat² (perkiraan)"

            total_valid_pixels = np.sum(~np.isnan(klasifikasi))

            if total_valid_pixels > 0 and pixel_area_sq_unit is not None:
                st.write(f"Total area terhitung di Kota Malang: **{(total_valid_pixels * pixel_area_sq_unit):.2f} {area_unit}**")

                class_counts = {
                    "Rawan Rendah": np.sum(klasifikasi == 1),
                    "Rawan Sedang": np.sum(klasifikasi == 2),
                    "Rawan Tinggi": np.sum(klasifikasi == 3)
                }

                st.subheader("Distribusi Area Rawan Banjir Berdasarkan Kategori")
                df_area_stats = pd.DataFrame(columns=["Kategori Rawan Banjir", f"Area ({area_unit})", "Persentase (%)"])

                for class_name, count in class_counts.items():
                    area_val = count * pixel_area_sq_unit
                    percentage = (count / total_valid_pixels) * 100 if total_valid_pixels > 0 else 0
                    df_area_stats = pd.concat([df_area_stats, pd.DataFrame([{"Kategori Rawan Banjir": class_name,
                                                                              f"Area ({area_unit})": f"{area_val:.2f}",
                                                                              "Persentase (%)": f"{percentage:.1f}"}])],
                                                ignore_index=True)
                st.table(df_area_stats)
            else:
                st.info("Tidak ada data klasifikasi yang valid untuk menghitung statistik area.")
        except Exception as e:
            st.error(f"Error menghitung statistik area rawan banjir: {e}")
    else:
        st.info("Informasi ukuran piksel tidak tersedia atau data klasifikasi kosong.")

    st.markdown("---")
    st.subheader("Statistik Jalur Sungai")
    if gdf_rivers is not None:
        try:
            # Reproject for accurate length calculation
            target_crs_proj = "EPSG:32749" # A common UTM zone for Indonesia, adjust if needed
            if gdf_rivers.crs != target_crs_proj:
                gdf_rivers_stats = gdf_rivers.to_crs(target_crs_proj)
            else:
                gdf_rivers_stats = gdf_rivers

            if len(gdf_rivers_stats) > 0:
                lengths = gdf_rivers_stats.geometry.length
                total_length = lengths.sum() / 1000
                avg_length = lengths.mean() / 1000
                max_length = lengths.max() / 1000

                st.write(f"Jumlah segmen sungai yang teridentifikasi di area Kota Malang: **{len(gdf_rivers_stats)}** segmen.")
                st.write(f"Total panjang seluruh jalur sungai: **{total_length:.2f} km**.")
                st.write(f"Panjang rata-rata per segmen sungai: **{avg_length:.2f} km**.")
                st.write(f"Panjang segmen sungai terpanjang: **{max_length:.2f} km**.")

                possible_name_columns = ['NAMA', 'NAME', 'REMARK', 'NAMOBJ', 'RIVER_NAME', 'NAMA_SUNGAI']
                nama_kolom = None
                for col in possible_name_columns:
                    if col in gdf_rivers_stats.columns:
                        nama_kolom = col
                        break

                if nama_kolom:
                    unique_names = gdf_rivers_stats[nama_kolom].dropna().unique()
                    if len(unique_names) > 0:
                        names_to_display = sorted([name for name in unique_names if pd.notna(name) and str(name).strip()])
                        if len(names_to_display) > 10:
                            st.write(f"Beberapa nama sungai yang teridentifikasi (10 teratas): **{', '.join(names_to_display[:10])}**.")
                            st.info(f"Ada total **{len(names_to_display)}** nama sungai unik yang ditemukan di Kota Malang.")
                        else:
                            st.write(f"Nama-nama sungai yang teridentifikasi: **{', '.join(names_to_display)}**.")
                    else:
                        st.info("Tidak ada nama sungai yang tersedia dalam data yang dimuat.")
            else:
                st.info("Tidak ada sungai yang berada dalam area Malang untuk dihitung statistiknya.")

        except Exception as e:
            st.error(f"Error menghitung statistik sungai: {e}")
    else:
        st.info("Data sungai tidak tersedia atau gagal dimuat.")

else:
    st.error("Gagal memuat atau memproses data. Harap periksa path file dan integritas data.")