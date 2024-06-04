#!/bin/sh

jupyter nbconvert --to script notebooks/* 

mv -f notebooks/*.py src/