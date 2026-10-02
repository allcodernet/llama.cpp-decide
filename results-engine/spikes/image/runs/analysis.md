## p(cat) per image (image-min-tokens 1024 unless noted)

| image | truth | chat p_yes | decide p_true (cat only) | /completion replay | decide p_true (4 fields) | image cells / positions | chat p_yes, default tokens | decide p_true, default tokens | cells / positions, default |
|---|---|---|---|---|---|---|---|---|---|
| cat03 | cat | 0.9990 | 0.9999 | 0.9999 | 1.0000 | 1602 / 42 | 0.9991 | 0.9999 | 1602 / 42 |
| kitten | cat | 0.9842 | 0.9998 | 0.9998 | 1.0000 | 1082 / 42 | 0.9867 | 0.9998 | 1082 / 42 |
| coco_39769 | cat | 0.9986 | 0.9996 | 0.9996 | 1.0000 | 1038 / 39 | 1.0000 | 0.9999 | 302 / 22 |
| wildcat | cat | 0.9985 | 0.9999 | 0.9999 | 1.0000 | 1842 / 48 | 0.9978 | 0.9999 | 1842 / 48 |
| lynx | cat | 0.9922 | 0.8383 | 0.8226 | 0.6576 | 1082 / 42 | 0.9782 | 0.9505 | 427 / 27 |
| fox | no cat | 0.0185 | 0.0012 | 0.0011 | 0.0000 | 1202 / 42 | 0.0126 | 0.0008 | 1202 / 42 |
| labrador | no cat | 0.0038 | 0.0001 | 0.0001 | 0.0000 | 1082 / 38 | 0.0349 | 0.0001 | 398 / 24 |
| coco_776 | no cat | 0.0004 | 0.0004 | 0.0003 | 0.0000 | 1082 / 42 | 0.0277 | 0.0001 | 262 / 22 |
| coco_139 | no cat | 0.0762 | 0.0716 | 0.0617 | 0.0287 | 1082 / 42 | 0.3368 | 0.0312 | 262 / 22 |
| coco_632 | no cat | 0.0185 | 0.0046 | 0.0057 | 0.0009 | 1038 / 39 | 0.0818 | 0.0048 | 302 / 22 |

## engine vs /completion replay (reference statistic)

| run | field | median abs diff | max abs diff | argmax agreement | disagreements over margin | pass |
|---|---|---|---|---|---|---|
| cat only, 1024 | cat | 9.4e-05 | 0.0157 | 1.00 | 0 | PASS |
| cat only, default tokens | cat | 3.6e-05 | 0.0177 | 1.00 | 0 | PASS |
| 4 fields, 1024 | cat | 2.1e-05 | 0.0205 | 1.00 | 0 | PASS |
| 4 fields, 1024 | dog | 0.00067 | 0.0090 | 1.00 | 0 | PASS |
| 4 fields, 1024 | animal | 6.2e-05 | 0.0208 | 1.00 | 0 | PASS |
| 4 fields, 1024 | indoors | 0.0013 | 0.0056 | 1.00 | 0 | PASS |

## multi-field answers (4 fields, 1024)

| image | cat | dog | animal | indoors |
|---|---|---|---|---|
| cat03 | 1.0000 | 0.0014 | cat (0.998) | 0.1024 |
| kitten | 1.0000 | 0.0028 | cat (0.999) | 0.9358 |
| coco_39769 | 1.0000 | 0.0012 | cat (0.998) | 0.9931 |
| wildcat | 1.0000 | 0.0019 | cat (0.980) | 0.8776 |
| lynx | 0.6576 | 0.0103 | cat (0.836) | 0.0135 |
| fox | 0.0000 | 0.0228 | fox (1.000) | 0.0027 |
| labrador | 0.0000 | 0.9973 | dog (1.000) | 0.0020 |
| coco_776 | 0.0000 | 0.0581 | other (0.958) | 0.9859 |
| coco_139 | 0.0287 | 0.0733 | none (0.906) | 0.9968 |
| coco_632 | 0.0009 | 0.0041 | none (0.988) | 0.9968 |

## timings per request with one image state (ms; medians over the 10 images)

| run | image cells (median) | encode | image decode | prefill (incl. both) | scoring | total |
|---|---|---|---|---|---|---|
| cat only, 1024 | 1082 | 324 | 853 | 1231 | 36.3 | 1286 |
| cat only, default tokens | 412 | 88 | 361 | 504 | 36.2 | 550 |
| 4 fields, 1024 | 1082 | 326 | 866 | 1248 | 82.8 | 1351 |

chat reference wall per image (one token, includes HTTP + base64): median 1339 ms (1024), 614 ms (default)

3 image states in one request (4 fields): total 4708 ms, encode 1321, image decode 3114, scoring 146, decodes 11; singles total 4870 ms; batch vs single max abs diff 0.0073, 0.00022, 0.00099
mixed request (text + image state): text state vs the same text alone max abs diff 0.00706
repeat cat03 x5 p_true: 0.999862, 0.999894, 0.999894, 0.999894, 0.999894; total ms: 1998, 2008, 2002, 2001, 2011
