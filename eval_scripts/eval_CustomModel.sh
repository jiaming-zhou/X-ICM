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



### bash eval_CustomModel.sh seeds episodes gpu_ids
### bash eval_scripts/eval_CustomModel.sh "0,99" 25 0,1

seeds=(${1//,/ })
episodes=$2
gpu_id=$3

echo "evaluate on 23 unseen tasks: ${unseen_tasks[@]}, using custom model"

tasks=("${unseen_tasks[@]}")
tasks_string=$(printf "%s," "${tasks[@]}")
tasks_string=${tasks_string%,}

method="CustomModel_Cross.ZS_test"

for seed in "${seeds[@]}"; do
    echo "eval using ${seed}..."
    CUDA_VISIBLE_DEVICES=$gpu_id xvfb-run -a python main_custom.py \
        "method.name=${method}" \
        "rlbench.tasks=[$tasks_string]" \
        "framework.start_seed=${seed}" \
        "framework.eval_episodes=${episodes}"
done
