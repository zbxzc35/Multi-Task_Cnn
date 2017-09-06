source ./env.sh
. ./script/generate_pair_image.sh
. ./script/generate_pair_sku.sh

function run {
    log "TRACE" "Start"
    day=`date -d "1 days ago" +%Y-%m-%d`
    
    # ----------------------generate_pair_sku start--------------------------------
    start_time=`date +%s`
    log "TRACE" "Start of generate_pair_sku."
    raw_input_path="/user/jd_ad/ads_reco/wangjincheng/sku_text_info"
    raw_save_path="${HADOOP_PROJECT_HOME}/pairwise_sku_all/${day}"
    
    generate_pair_sku ${raw_input_path} ${raw_save_path}
    
    if [ $? -ne 0 ]; then
        log "ERROR" "fail to run generate_pair_sku"
        exit
    else
        end_time=`date +%s`
        log "TRACE" "End of generate_pair_sku, it takes $(($end_time - $start_time)) seconds."
    fi
   # ----------------------generate_pair_sku end---------------------------

    # ----------------------generate_pair_image start--------------------------------
    start_time=`date +%s`
    log "TRACE" "Start of generate_pair_image."
    input_path="${raw_save_path}"
    save_path="${HADOOP_PROJECT_HOME}/pairwise_image_all/${day}"
    job_name="generate_pair_image_${day}"
    
    generate_pair_image ${day} ${input_path} ${save_path} ${job_name}
    
    if [ $? -ne 0 ]; then
        log "ERROR" "fail to run generate_pair_image"
        exit
    else
        end_time=`date +%s`
        log "TRACE" "End of generate_pair_image, it takes $(($end_time - $start_time)) seconds."
    fi
    # ----------------------generate_pair_image end---------------------------
    log "TRACE" "Finish!"
}
                                 
run 2>&1 | tee $LOG_FILE
