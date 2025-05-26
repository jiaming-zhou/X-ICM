### 18 seen tasks
rlbench_tasks=(
    "close_jar" 
    "push_buttons" 
    "stack_blocks" 
    "open_drawer" 
    "sweep_to_dustpan_of_size" 
    "put_item_in_drawer" 
    "reach_and_drag" 
    "turn_tap" 
    "slide_block_to_color_target" 
    "put_groceries_in_cupboard" 
    "place_shape_in_shape_sorter" 
    "put_money_in_safe" 
    "place_cups" 
    "place_wine_at_rack_location" 
    "light_bulb_in" 
    "insert_onto_square_peg" 
    "meat_off_grill" 
    "stack_cups"
    )

### 23 unseen tasks
unseen_tasks=( 
    "put_toilet_roll_on_stand" 
    "put_knife_on_chopping_board"
    "close_fridge"
    "close_microwave" 
    "close_laptop_lid"
    "phone_on_base" 
    "toilet_seat_down" 
    "lamp_off"
    "lamp_on"
    "put_books_on_bookshelf" 
    "put_umbrella_in_umbrella_stand" 
    "open_grill" 
    "put_rubbish_in_bin" 
    "take_usb_out_of_computer" 
    "take_lid_off_saucepan"
    "take_plate_off_colored_dish_rack" 
    "basketball_in_hoop" 
    "scoop_with_spatula" 
    "straighten_rope"  
    "turn_oven_on" 
    "beat_the_buzz" 
	"water_plants" 
    "unplug_charger" 
)



### bash eval_XICM.sh seeds episodes modelname num_icls gpu_ids ranking_method
### bash eval_scripts/eval_XICM.sh "0,25,50,75,99" 25 Qwen2.5.7B.instruct 1 0,1,2,3,4,5,6,7 "random"
### bash eval_scripts/eval_XICM.sh "0,25,50,75,99" 25 Qwen2.5.7B.instruct 1 0,1,2,3,4,5,6,7 "lang_vis.out"


seeds=(${1//,/ })
episodes=$2
modelname=$3
demo_num_per_icl=$4
gpu_id=$5
ranking_method=$6

echo "evaluate on 23 unseen tasks: ${unseen_tasks[@]}, using ${ranking_method} ranking method"

tasks=("${unseen_tasks[@]}")
tasks_string=$(printf "%s," "${tasks[@]}")
tasks_string=${tasks_string%,}

method="XICM_Cross.ZS_Ranking.${ranking_method}_${modelname}_icl.${demo_num_per_icl}_test"

for seed in "${seeds[@]}"; do
    echo "eval using ${seed}..."
    CUDA_VISIBLE_DEVICES=$gpu_id xvfb-run -a python main.py \
        "method.name=${method}" \
        "rlbench.tasks=[$tasks_string]" \
        "framework.start_seed=${seed}" \
        "framework.eval_episodes=${episodes}" \
        "framework.demo_num_per_icl=${demo_num_per_icl}" \
        "framework.ranking_method=${ranking_method}"
done
