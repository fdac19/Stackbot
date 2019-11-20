#!/bin/bash
cat NEEDED_LIBRARIES | while read line || [[ -n $line ]];
do
  cmd="sudo -H pip3 install "$line
  eval $cmd
done
