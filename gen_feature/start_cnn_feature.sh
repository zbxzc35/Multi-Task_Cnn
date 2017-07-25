source ./env.sh
. ./script/generate_image_feature.sh
. ./script/generate_train_data_attr.sh
. ./script/generate_label.sh

function run {
    log "TRACE" "Start"

#    day=`date -d "1 days ago" +%Y-%m-%d`
    day="2017-07-18"
    # ----------------------generate_image_feature start----------------------
    start_time=`date +%s`
    log "TRACE" "Start of generate_image_feature."
    raw_input_path="/user/jd_ad/ouhuisi/sku_redis/2017-04-12/sku_attr"
    raw_save_path="${HADOOP_PROJECT_HOME}/raw_feature/${day}"
    job_name="generate_image_feature_${day}"

#    generate_image_feature ${day} ${raw_input_path} ${raw_save_path} $job_name
#
#    if [ $? -ne 0 ]; then
#      log "ERROR" "fail to run generate_image_feature"
#      exit
#    else
#      end_time=`date +%s`
#      log "TRACE" "End of generate_image_feature, it takes $(($end_time - $start_time)) seconds."  
#    fi
#
    # ----------------------generate_train_data_attr start----------------------
    start_time=`date +%s`
    log "TRACE" "Start of generate_train_data_attr."
    train_attr_input_path=${raw_save_path}
    train_attr_save_path="${HADOOP_PROJECT_HOME}/train_data_attr/${day}"
    job_name="generate_train_data_attr_${day}"
 
    generate_train_data_attr ${train_attr_input_path} ${train_attr_save_path} $job_name
 
    if [ $? -ne 0 ]; then
      log "ERROR" "fail to run generate_train_data_attr"
      exit
    else
      end_time=`date +%s`
      log "TRACE" "End of generate_train_data_attr, it takes $(($end_time - $start_time)) seconds."
    fi
    # ----------------------generate_train_data end---------------------- 
    
#    # ----------------------generate_label start----------------------
#    start_time=`date +%s`
#    log "TRACE" "Start of generate_label."
#    label_input_path=${raw_save_path}
#    label_save_path="${HADOOP_PROJECT_HOME}/all_labels/${day}"
#    job_name="generate_label_${day}"
#
#    generate_label ${label_input_path} ${label_save_path} $job_name
# 
#    if [ $? -ne 0 ]; then
#      log "ERROR" "fail to run generate_label"
#      exit
#    else
#      end_time=`date +%s`
#      log "TRACE" "End of generate_label, it takes $(($end_time - $start_time)) seconds."
#    fi
    # ----------------------generate_label end---------------------- 
    log "TRACE" "Finish!"
}

run 2>&1 | tee $LOG_FILE 
