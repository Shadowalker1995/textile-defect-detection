#!/bin/bash

echo ===========
echo "2-way 4-shot"
echo ===========
# python experiments/proto_nets_test.py --dataset Fabric -rs 42 --n-train 4 --n-test 4 --k-train 2 --k-test 2 --q-train 12 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 2021 --n-train 4 --n-test 4 --k-train 2 --k-test 2 --q-train 12 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 369 --n-train 4 --n-test 4 --k-train 2 --k-test 2 --q-train 12 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 218 --n-train 4 --n-test 4 --k-train 2 --k-test 2 --q-train 12 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 1995 --n-train 4 --n-test 4 --k-train 2 --k-test 2 --q-train 12 --q-test 1

echo ===========
echo "2-way 1-shot"
echo ===========
# python experiments/proto_nets_test.py --dataset Fabric -rs 42 --n-train 1 --n-test 1 --k-train 2 --k-test 2 --q-train 12 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 2021 --n-train 1 --n-test 1 --k-train 2 --k-test 2 --q-train 12 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 369 --n-train 1 --n-test 1 --k-train 2 --k-test 2 --q-train 12 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 218 --n-train 1 --n-test 1 --k-train 2 --k-test 2 --q-train 12 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 1995 --n-train 1 --n-test 1 --k-train 2 --k-test 2 --q-train 12 --q-test 1

echo ===========
echo "4-way 4-shot"
echo ===========
python experiments/proto_nets_test.py --dataset Fabric -rs 42 --n-train 4 --n-test 4 --k-train 4 --k-test 4 --q-train 12 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 2021 --n-train 4 --n-test 4 --k-train 4 --k-test 4 --q-train 12 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 369 --n-train 4 --n-test 4 --k-train 4 --k-test 4 --q-train 12 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 218 --n-train 4 --n-test 4 --k-train 4 --k-test 4 --q-train 12 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 1995 --n-train 4 --n-test 4 --k-train 4 --k-test 4 --q-train 12 --q-test 1

echo ===========
echo "4-way 1-shot"
echo ===========
python experiments/proto_nets_test.py --dataset Fabric -rs 42 --n-train 1 --n-test 1 --k-train 4 --k-test 4 --q-train 12 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 2021 --n-train 1 --n-test 1 --k-train 4 --k-test 4 --q-train 12 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 369 --n-train 1 --n-test 1 --k-train 4 --k-test 4 --q-train 12 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 218 --n-train 1 --n-test 1 --k-train 4 --k-test 4 --q-train 12 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 1995 --n-train 1 --n-test 1 --k-train 4 --k-test 4 --q-train 12 --q-test 1

echo ===========
echo "6-way 4-shot"
echo ===========
python experiments/proto_nets_test.py --dataset Fabric -rs 42 --n-train 4 --n-test 4 --k-train 6 --k-test 6 --q-train 6 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 2021 --n-train 4 --n-test 4 --k-train 6 --k-test 6 --q-train 6 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 369 --n-train 4 --n-test 4 --k-train 6 --k-test 6 --q-train 6 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 218 --n-train 4 --n-test 4 --k-train 6 --k-test 6 --q-train 6 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 1995 --n-train 4 --n-test 4 --k-train 6 --k-test 6 --q-train 6 --q-test 1

echo ===========
echo "6-way 1-shot"
echo ===========
python experiments/proto_nets_test.py --dataset Fabric -rs 42 --n-train 1 --n-test 1 --k-train 6 --k-test 6 --q-train 6 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 2021 --n-train 1 --n-test 1 --k-train 6 --k-test 6 --q-train 6 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 369 --n-train 1 --n-test 1 --k-train 6 --k-test 6 --q-train 6 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 218 --n-train 1 --n-test 1 --k-train 6 --k-test 6 --q-train 6 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 1995 --n-train 1 --n-test 1 --k-train 6 --k-test 6 --q-train 6 --q-test 1

echo ===========
echo "8-way 4-shot"
echo ===========
python experiments/proto_nets_test.py --dataset Fabric -rs 42 --n-train 4 --n-test 4 --k-train 8 --k-test 8 --q-train 4 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 2021 --n-train 4 --n-test 4 --k-train 8 --k-test 8 --q-train 4 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 369 --n-train 4 --n-test 4 --k-train 8 --k-test 8 --q-train 4 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 218 --n-train 4 --n-test 4 --k-train 8 --k-test 8 --q-train 4 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 1995 --n-train 4 --n-test 4 --k-train 8 --k-test 8 --q-train 4 --q-test 1

echo ===========
echo "8-way 1-shot"
echo ===========
python experiments/proto_nets_test.py --dataset Fabric -rs 42 --n-train 1 --n-test 1 --k-train 8 --k-test 8 --q-train 4 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 2021 --n-train 1 --n-test 1 --k-train 8 --k-test 8 --q-train 4 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 369 --n-train 1 --n-test 1 --k-train 8 --k-test 8 --q-train 4 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 218 --n-train 1 --n-test 1 --k-train 8 --k-test 8 --q-train 4 --q-test 1
python experiments/proto_nets_test.py --dataset Fabric -rs 1995 --n-train 1 --n-test 1 --k-train 8 --k-test 8 --q-train 4 --q-test 1
