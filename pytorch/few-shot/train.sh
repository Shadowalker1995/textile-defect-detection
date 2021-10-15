#!/bin/bash

# echo ===========
# echo "2-way 4-shot"
# echo ===========
# python experiments/proto_nets.py --dataset Fabric --n-train 4 --n-test 4 --k-train 2 --k-test 2 --q-train 12 --q-test 1
# {'test_loss': 0.12131671823653083, 'test_acc': 0.9604}
# {'val_loss': 0.15634882621215107, 'val_acc': 0.9473}
# python experiments/proto_nets_test.py --dataset Fabric --n-train 4 --n-test 4 --k-train 2 --k-test 2 --q-train 12 --q-test 1

# echo ===========
# echo "2-way 1-shot"
# echo ===========
# python experiments/proto_nets.py --dataset Fabric --n-train 1 --n-test 1 --k-train 2 --k-test 2 --q-train 12 --q-test 1
# {'test_loss': 0.16853973467004113, 'test_acc': 0.933}
# {'val_loss': 0.18849388689147525, 'val_acc': 0.9231}
# python experiments/proto_nets_test.py --dataset Fabric --n-train 1 --n-test 1 --k-train 2 --k-test 2 --q-train 12 --q-test 1

# echo ===========
# echo "3-way 4-shot"
# echo ===========
# python experiments/proto_nets.py --dataset Fabric --n-train 4 --n-test 4 --k-train 3 --k-test 3 --q-train 12 --q-test 1

# echo ===========
# echo "3-way 1-shot"
# echo ===========
# python experiments/proto_nets.py --dataset Fabric --n-train 1 --n-test 1 --k-train 3 --k-test 3 --q-train 12 --q-test 1
# {'test_loss': 0.5239135518169453, 'test_acc': 0.8835}
# {'val_loss': 0.5924311390482466, 'val_acc': 0.8611}}
# python experiments/proto_nets_test.py --dataset Fabric --n-train 1 --n-test 1 --k-train 3 --k-test 3 --q-train 12 --q-test 1

# echo ===========
# echo "4-way 4-shot"
# echo ===========
# python experiments/proto_nets.py --dataset Fabric --n-train 4 --n-test 4 --k-train 4 --k-test 4 --q-train 12 --q-test 1

# echo ===========
# echo "4-way 1-shot"
# echo ===========
# python experiments/proto_nets.py --dataset Fabric --n-train 1 --n-test 1 --k-train 4 --k-test 4 --q-train 12 --q-test 1
# python experiments/proto_nets_test.py --dataset Fabric --n-train 1 --n-test 1 --k-train 4 --k-test 4 --q-train 12 --q-test 1

# echo ===========
# echo "6-way 4-shot"
# echo ===========
# python experiments/proto_nets.py --dataset Fabric --n-train 4 --n-test 4 --k-train 6 --k-test 6 --q-train 6 --q-test 1
#
# echo ===========
# echo "6-way 1-shot"
# echo ===========
# python experiments/proto_nets.py --dataset Fabric --n-train 1 --n-test 1 --k-train 6 --k-test 6 --q-train 6 --q-test 1
#
echo ===========
echo "8-way 3-shot"
echo ===========
python experiments/proto_nets.py --dataset Fabric --n-train 3 --n-test 3 --k-train 8 --k-test 8 --q-train 4 --q-test 1
#
# echo ===========
# echo "8-way 1-shot"
# echo ===========
# python experiments/proto_nets.py --dataset Fabric --n-train 1 --n-test 1 --k-train 8 --k-test 8 --q-train 4 --q-test 1


