#!/bin/bash



if [[ $1 == 'tnews' ]]; then
  python tnews.py -c few_all
  python tnews.py -c 0
  python tnews.py -c 1
  python tnews.py -c 2
  python tnews.py -c 3
  python tnews.py -c 4

else

  echo 'unknown argment 1'
fi
