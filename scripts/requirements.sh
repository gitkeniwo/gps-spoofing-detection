#!/bin/bash
pip install pipreqs -q
pipreqs --scan-notebooks --savepath ./requirements.txt notebooks/ 

