#!/bin/sh

jupyter nbconvert --to script notebooks/* 

mv notebooks/*.py src/