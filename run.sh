#!/bin/bash

if [[ $1 == 'eprstmt' ]]; then
  python eprstmt.py -c 0
  python eprstmt.py -c 1
  python eprstmt.py -c 2
  python eprstmt.py -c 3
  python eprstmt.py -c 4
  python eprstmt.py -c few_all

elif [[ $1 == 'tnews' ]]; then
  python tnews.py -c few_all
  python tnews.py -c 0
  python tnews.py -c 1
  python tnews.py -c 2
  python tnews.py -c 3
  python tnews.py -c 4

elif [[ $1 == 'csldcp' ]]; then
  python csldcp.py -c 0
  python csldcp.py -c 1
  python csldcp.py -c 2
  python csldcp.py -c 3
  python csldcp.py -c 4
  python csldcp.py -c few_all

elif [[ $1 == 'iflytek' ]]; then
  python iflytek.py -c 0
  python iflytek.py -c 1
  python iflytek.py -c 2
  python iflytek.py -c 3
  python iflytek.py -c 4
  python iflytek.py -c few_all

elif [[ $1 == 'csl' ]]; then
  python csl.py -c few_all
  python csl.py -c 0
  python csl.py -c 1
  python csl.py -c 2
  python csl.py -c 3
  python csl.py -c 4

elif [[ $1 == 'cluewsc' ]]; then
  python cluewsc.py -c few_all
  python cluewsc.py -c 0
  python cluewsc.py -c 1
  python cluewsc.py -c 2
  python cluewsc.py -c 3
  python cluewsc.py -c 4

elif [[ $1 == 'bustm' ]]; then
  python bustm.py -c 0
  python bustm.py -c 1
  python bustm.py -c 2
  python bustm.py -c 3
  python bustm.py -c 4
  python bustm.py -c few_all

elif [[ $1 == 'ocnli' ]]; then
  python ocnli.py -c few_all
  python ocnli.py -c 0
  python ocnli.py -c 1
  python ocnli.py -c 2
  python ocnli.py -c 3
  python ocnli.py -c 4

else
  echo 'unknown argment 1'
fi
