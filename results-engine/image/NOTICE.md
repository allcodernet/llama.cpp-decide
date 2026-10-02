# Images used by the sub-project 4 results: notice

No image is in this repository. Every image comes from COCO 2017 validation (COCO Consortium, https://cocodataset.org;
annotations licensed under CC BY 4.0, https://creativecommons.org/licenses/by/4.0/) and is downloaded into the
git-ignored `local/` directory. Each image keeps the licence of its Flickr source, named in the annotations
(`instances_val2017.json`) by its `license` id.

## Integrity images

`states-integrity.jsonl`, the smoke and the live checks use four images, checked by `integrity-images.sha256` and
fetched into `local/image/integrity/coco-<id>.jpg` from their `coco_url` (Task 0). `coco-000000039769.png`
is a PNG copy of image 39769 (its decoded pixels saved as PNG with Pillow), under the same licence.

| image id | COCO URL | Flickr URL | licence id | licence |
|---|---|---|---|---|
| 39769 | http://images.cocodataset.org/val2017/000000039769.jpg | http://farm1.staticflickr.com/60/210383891_f91a89fd5e_z.jpg | 5 | Attribution-ShareAlike License, http://creativecommons.org/licenses/by-sa/2.0/ |
| 776 | http://images.cocodataset.org/val2017/000000000776.jpg | http://farm2.staticflickr.com/1399/1488578517_9c6bfc45de_z.jpg | 1 | Attribution-NonCommercial-ShareAlike License, http://creativecommons.org/licenses/by-nc-sa/2.0/ |
| 139 | http://images.cocodataset.org/val2017/000000000139.jpg | http://farm9.staticflickr.com/8035/8024364858_9c41dc1666_z.jpg | 2 | Attribution-NonCommercial License, http://creativecommons.org/licenses/by-nc/2.0/ |
| 632 | http://images.cocodataset.org/val2017/000000000632.jpg | http://farm2.staticflickr.com/1241/1243324748_eea455da9f_z.jpg | 3 | Attribution-NonCommercial-NoDerivs License, http://creativecommons.org/licenses/by-nc-nd/2.0/ |

## COCO evaluation subset

The 200 images of the evaluation (spec 9.6): `coco/NOTICE.md` (written by `coco.py select`), with the licence ids of
`coco/subset.jsonl`.
