#!/bin/bash

for script in $(ls scripts); do
	bash scripts/$script
done
