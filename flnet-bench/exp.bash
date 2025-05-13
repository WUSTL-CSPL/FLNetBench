#!/bin/bash

# for i in {1..5}; do
#     config_file="configs/uplink_real${i}.json"
#     python run.py --config "$config_file"
# done

# config_file="configs/uplink_real1.json"
# python run.py --config "$config_file"

# mkdir test_cnn_baseline
# mv *.png test_cnn_baseline/
# mv *.csv test_cnn_baseline/test_cnn_baseline.csv
# mv *.pkl test_cnn_baseline/
# mv *.pdf test_cnn_baseline/

# mkdir sync_baseline_noniid
# mv *.png sync_baseline_noniid/
# mv *.csv sync_baseline_noniid/
# mv *.pkl sync_baseline_noniid/
# mv *.pdf sync_baseline_noniid/

# mkdir sync_uplink_staleness_10
# mv *.png sync_uplink_staleness_10/
# mv *.csv sync_uplink_staleness_10/
# mv *.pkl sync_uplink_staleness_10/
# mv *.pdf sync_uplink_staleness_10/

# mkdir sync_uplink_staleness_50_noniid
# mv *.png sync_uplink_staleness_50_noniid/
# mv *.csv sync_uplink_staleness_50_noniid/
# mv *.pkl sync_uplink_staleness_50_noniid/
# mv *.pdf sync_uplink_staleness_50_noniid/

# mkdir sync_downlink_staleness_50
# mv *.png sync_downlink_staleness_50/
# mv *.csv sync_downlink_staleness_50/
# mv *.pkl sync_downlink_staleness_50/
# mv *.pdf sync_downlink_staleness_50/

# mkdir sync_downlink_staleness_50_noniid
# mv *.png sync_downlink_staleness_50_noniid/
# mv *.csv sync_downlink_staleness_50_noniid/
# mv *.pkl sync_downlink_staleness_50_noniid/
# mv *.pdf sync_downlink_staleness_50_noniid/

# mkdir sync_missing_50
# mv *.png sync_missing_50/
# mv *.csv sync_missing_50/
# mv *.pkl sync_missing_50/
# mv *.pdf sync_missing_50/

# mkdir sync_missing_50_noniid
# mv *.png sync_missing_50_noniid/
# mv *.csv sync_missing_50_noniid/
# mv *.pkl sync_missing_50_noniid/
# mv *.pdf sync_missing_50_noniid/

# mkdir sync_mlp_baseline
# mv *.png sync_mlp_baseline/
# mv *.csv sync_mlp_baseline/
# mv *.pkl sync_mlp_baseline/
# mv *.pdf sync_mlp_baseline/

# mkdir sync_uplink_staleness_50_mlp
# mv *.png sync_uplink_staleness_50_mlp/
# mv *.csv sync_uplink_staleness_50_mlp/
# mv *.pkl sync_uplink_staleness_50_mlp/
# mv *.pdf sync_uplink_staleness_50_mlp/

# mkdir sync_downlink_staleness_50_mlp
# mv *.png sync_downlink_staleness_50_mlp/
# mv *.csv sync_downlink_staleness_50_mlp/
# mv *.pkl sync_downlink_staleness_50_mlp/
# mv *.pdf sync_downlink_staleness_50_mlp/

# mkdir sync_missing_50_mlp
# mv *.png sync_missing_50_mlp/
# mv *.csv sync_missing_50_mlp/
# mv *.pkl sync_missing_50_mlp/
# mv *.pdf sync_missing_50_mlp/

# mkdir sync_mnist_baseline
# mv *.png sync_mnist_baseline/
# mv *.csv sync_mnist_baseline/
# mv *.pkl sync_mnist_baseline/
# mv *.pdf sync_mnist_baseline/

# mkdir sync_rnn_baseline
# mv *.png sync_rnn_baseline/
# mv *.csv sync_rnn_baseline/
# mv *.pkl sync_rnn_baseline/
# mv *.pdf sync_rnn_baseline/

# mkdir sync_uplink_staleness_50_rnn
# mv *.png sync_uplink_staleness_50_rnn/
# mv *.csv sync_uplink_staleness_50_rnn/
# mv *.pkl sync_uplink_staleness_50_rnn/
# mv *.pdf sync_uplink_staleness_50_rnn/

# mkdir sync_downlink_staleness_50_rnn
# mv *.png sync_downlink_staleness_50_rnn/
# mv *.csv sync_downlink_staleness_50_rnn/
# mv *.pkl sync_downlink_staleness_50_rnn/
# mv *.pdf sync_downlink_staleness_50_rnn/

# mkdir sync_missing_50_rnn
# mv *.png sync_missing_50_rnn/
# mv *.csv sync_missing_50_rnn/
# mv *.pkl sync_missing_50_rnn/
# mv *.pdf sync_missing_50_rnn/

# mkdir video_baseline
# mv *.png video_baseline/
# mv *.csv video_baseline/
# mv *.pkl video_baseline/
# mv *.pdf video_baseline/

config_file="configs/uplink_restnet_async.json"
python run.py --config "$config_file"

mkdir async_missing_50/
mv *.png async_missing_50/
mv accuracy_data.csv async_missing_50/async_missing_50.csv
mv *.csv async_missing_50/
mv *.pkl async_missing_50/
mv *.pdf async_missing_50/
mv *.txt async_missing_50/

# config_file="configs/uplink_restnet.json"
# python run.py --config "$config_file"

# mkdir resnet_epoch5/
# mv *.png resnet_epoch5/
# mv accuracy_data.csv resnet_epoch5/resnet_epoch5.csv
# mv *.csv resnet_epoch5/
# mv *.pkl resnet_epoch5/
# mv *.pdf resnet_epoch5/

# mkdir resnet_uplink_20
# mv *.png resnet_uplink_20/
# mv *.csv resnet_uplink_20/resnet_uplink_20.csv
# mv *.pkl resnet_uplink_20/
# mv *.pdf resnet_uplink_20/

# mkdir cifar_cnn_baseline
# mv *.png cifar_cnn_baseline/
# mv *.csv cifar_cnn_baseline/cifar_cnn_baseline.csv
# mv *.pkl cifar_cnn_baseline/
# mv *.pdf cifar_cnn_baseline/

# mkdir resnet_uplink_50
# mv *.png resnet_uplink_50/
# mv *.csv resnet_uplink_50/
# mv *.pkl resnet_uplink_50/
# mv *.pdf resnet_uplink_50/

# python fed_avg.py --norm bn --partition iid --commrounds 100 --clientfr 0.1 --numclient 100 --clientepochs 20 --clientbs 64 --clientlr 0.0001 --alpha_partition 0.5

# config_file="configs/uplink_imdb.json"
# python run.py --config "$config_file"

# mkdir imdb_baseline/
# mv *.png imdb_baseline/
# mv *.csv imdb_baseline/
# mv *.pkl imdb_baseline/
# mv *.pdf imdb_baseline/