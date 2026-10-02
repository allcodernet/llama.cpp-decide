# COCO val2017 subset: notice

`subset.jsonl` is derived from the COCO 2017 validation annotations (`instances_val2017.json`) of the
COCO Consortium (https://cocodataset.org), licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
Changes: 200 images chosen with a fixed seed and one yes/no label per category derived from the instance
annotations (`results-engine/image/coco.py`, rule in spec 8 of docs/specs/2026-10-02-image-input-design.md).

The images are not in this repository: `coco.py download` fetches them from their `coco_url` into the
git-ignored `local/` directory. Each image keeps the licence of its Flickr source, named by its `license` id:

| licence id | name | URL | images |
|---|---|---|---|
| 1 | Attribution-NonCommercial-ShareAlike License | http://creativecommons.org/licenses/by-nc-sa/2.0/ | 51 |
| 2 | Attribution-NonCommercial License | http://creativecommons.org/licenses/by-nc/2.0/ | 26 |
| 3 | Attribution-NonCommercial-NoDerivs License | http://creativecommons.org/licenses/by-nc-nd/2.0/ | 54 |
| 4 | Attribution License | http://creativecommons.org/licenses/by/2.0/ | 35 |
| 5 | Attribution-ShareAlike License | http://creativecommons.org/licenses/by-sa/2.0/ | 20 |
| 6 | Attribution-NoDerivs License | http://creativecommons.org/licenses/by-nd/2.0/ | 14 |
