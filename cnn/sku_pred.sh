#!/bin/bash

test_pred_path='/user/jd_ad/ads_reco/xuzhexuan/multi-task_cnn/pred_data'
hadoop fs -mkdir ${test_pred_path}

for i in {0..999}
do
    test_out_path="../data/test_data"
    
    if [ $i -lt 10 ]; then
        test_input_path="/user/jd_ad/ads_reco/xuzhexuan/multi-task_cnn/pred_data_16attr/part-0000${i}"
        test_file="../data/test_data/part-0000${i}"
        echo ${test_input_path}
    elif [ $i -ge 10 ] && [ $i -lt 100 ]; then
        test_input_path="/user/jd_ad/ads_reco/xuzhexuan/multi-task_cnn/pred_data_16attr/part-000${i}"
        test_file="../data/test_data/part-000${i}"
        echo ${test_input_path}
    elif [ $i -ge 100 ] && [ $i -lt 1000 ]; then
        test_input_path="/user/jd_ad/ads_reco/xuzhexuan/multi-task_cnn/pred_data_16attr/part-00${i}"
        test_file="../data/test_data/part-00${i}"
        echo ${test_input_path}
    fi

    if [ -e "${test_file}" ]; then
        rm ${test_file}
    fi
    
    # 加载数据
    hadoop fs -get ${test_input_path} ${test_out_path}

    #将数据转成tfrecord
    test_tfrecord="../data/test${i}.bin"
    test_num=`python gen_test_tfrecord.py ${test_file} ${test_tfrecord}`

    test_pred="../data/test_pred/part${i}"
    python multi-task_cnn_pred.py ${test_num} ${test_tfrecord} ${test_pred}
    
    #把预测结果上传到hadoop上
    hadoop fs -put ${test_pred} ${test_pred_path}

    #删除相应的本地数据
    rm ${test_file}
    rm ${test_tfrecord}
#rm ${test_pred}
    
done
