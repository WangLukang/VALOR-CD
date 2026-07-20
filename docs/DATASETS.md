# Dataset Preparation

## WHU-CD

The WHU building change detection dataset contains co-registered aerial imagery from 2012 and 2016, together with a binary building-change raster. Obtain it from the [official WHU page](https://gpcv.whu.edu.cn/data/building_dataset.html) and follow its license and citation requirements.

The preparation script reads the large TIFFs window by window with Rasterio, so it does not load either 1.4 GB RGB image into memory at once.

- Source dimensions: 32,507 x 15,354 pixels.
- Tile size: 224 x 224 pixels.
- Stride: 224 pixels in both axes.
- Overlap: none.
- Border handling: incomplete right/bottom strips are discarded (27 columns and 122 rows for the released source files).
- Image-level label: `1` when the tile mask contains at least one changed pixel; otherwise `0`.
- Split: spatial bands along the row axis, 80%/10%/10%, with one ignored tile row at each split boundary.

Expected split statistics:

| Split | Total | Positive | Negative |
|---|---:|---:|---:|
| train | 7,685 | 1,707 | 5,978 |
| validation | 725 | 181 | 544 |
| test | 870 | 203 | 667 |
| ignored gap | 580 | 130 | 450 |

Run:

```bash
python tools/prepare_whu_dataset.py
```

Use `--overwrite` only when intentionally rebuilding `data/WHU_224`.

## LEVIR-CD

LEVIR-CD contains 637 pairs of 1024 x 1024 very-high-resolution images. Download it from the [official LEVIR-CD page](https://justchenhao.github.io/LEVIR/) and respect its academic-use terms.

- Tile size: 224 x 224 pixels.
- Default stride: 224 pixels.
- Border handling: a final boundary-aligned tile is added when needed, so neighboring tiles overlap only near the right/bottom edge.
- Image-level label: `1` when the tile mask contains at least one changed pixel; otherwise `0`.
- Split: the official train/validation/test assignment is preserved.

Expected split statistics:

| Split | Total | Positive | Negative |
|---|---:|---:|---:|
| train | 11,125 | 4,554 | 6,571 |
| validation | 1,600 | 636 | 964 |
| test | 3,200 | 1,372 | 1,828 |

Run:

```bash
python tools/prepare_levir_dataset.py
```

## Generated Layout

Both preparation tools produce the same loader-facing structure:

```text
data/<DATASET>_224/
  T1/
  T2/
  masks/
  tiles.csv
  manifests_*/
    train.csv
    val.csv
    test.csv
```

Pixel masks are read for validation/test metrics and checkpoint selection. Stage 1 training uses only the binary image-level `label` column from the training manifest.
