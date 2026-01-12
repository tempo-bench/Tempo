
python run.py \
    --dataset_dir my_ir_dataset \
    --model bm25 \
    --output_dir outputs \
    --encode_batch_size 1024 \
    --domains bitcoin cardano economics genealogy history hsm iota law monero politics quant  travel  workplace

python run.py \
    --dataset_dir ./my_ir_dataset \
    --model bge \
    --output_dir outputs \
    --encode_batch_size 1024 \
    --domains bitcoin cardano economics genealogy history hsm iota law monero politics quant  travel  workplace 

python run.py \
    --dataset_dir my_ir_dataset \
    --model contriever \
    --output_dir outputs \
    --encode_batch_size 1024 \
    --domains bitcoin cardano economics genealogy history hsm iota law monero politics quant  travel  workplace 

python run.py \
    --dataset_dir my_ir_dataset \
    --model diver-retriever \
    --output_dir outputs \
    --encode_batch_size 1024 \
    --domains bitcoin cardano economics genealogy history hsm iota law monero politics quant  travel  workplace 


python run.py \
    --dataset_dir my_ir_dataset \
    --model e5 \
    --output_dir outputs \
    --encode_batch_size 1024 \
    --domains bitcoin cardano economics genealogy history hsm iota law monero politics quant  travel  workplace 

python run.py \
    --dataset_dir my_ir_dataset \
    --model reasonir \
    --output_dir outputs \
    --encode_batch_size 1 \
     --model_data_dir queries \
    --domains  bitcoin cardano economics genealogy history hsm iota law monero politics quant  travel  workplace



python run.py \
    --dataset_dir my_ir_dataset \
    --model qwen \
    --output_dir outputs \
    --encode_batch_size 1024 \
    --domains bitcoin cardano economics genealogy history hsm iota law monero politics quant  travel  workplace

python run.py \
    --dataset_dir my_ir_dataset \
    --model qwen2 \
    --output_dir outputs \
    --encode_batch_size 1024 \
    --domains bitcoin cardano economics genealogy history hsm iota law monero politics quant  travel  workplace

python run.py \
    --dataset_dir my_ir_dataset \
    --model rader \
    --output_dir outputs \
    --encode_batch_size 1024 \
    --domains bitcoin cardano economics genealogy history hsm iota law monero politics quant  travel  workplace

python run.py \
    --dataset_dir my_ir_dataset \
    --model sf \
    --output_dir outputs \
    --encode_batch_size 1024 \
    --domains bitcoin cardano economics genealogy history hsm iota law monero politics quant  travel  workplace












