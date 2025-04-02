all: \
	index.md \
	0_intro.ipynb \
	1_create_server.ipynb \
	2_prepare_data.ipynb \
	3_launch_jupyter.ipynb \
	workspace/4_eval_offline.ipynb \
	5_delete.ipynb

clean: 
	rm index.md \
	0_intro.ipynb \
	1_create_server.ipynb \
	2_prepare_data.ipynb \
	3_launch_jupyter.ipynb \
	workspace/4_eval_offline.ipynb  \
	5_delete.ipynb

index.md: snippets/*.md 
	cat snippets/intro.md \
		snippets/create_server.md \
		snippets/prepare_data.md \
		snippets/launch_jupyter.md \
		snippets/eval_offline.md \
		snippets/delete.md \
		snippets/footer.md \
		> index.tmp.md
	grep -v '^:::' index.tmp.md > index.md
	rm index.tmp.md
	cat snippets/footer.md >> index.md

0_intro.ipynb: snippets/intro.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/intro.md \
                -o 0_intro.ipynb  
	sed -i 's/attachment://g' 0_intro.ipynb


1_create_server.ipynb: snippets/create_server.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/create_server.md \
                -o 1_create_server.ipynb  
	sed -i 's/attachment://g' 1_create_server.ipynb

2_prepare_data.ipynb: snippets/prepare_data.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/prepare_data.md \
                -o 2_prepare_data.ipynb  
	sed -i 's/attachment://g' 2_prepare_data.ipynb

3_launch_jupyter.ipynb: snippets/launch_jupyter.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/launch_jupyter.md \
                -o 3_launch_jupyter.ipynb  
	sed -i 's/attachment://g' 3_launch_jupyter.ipynb


workspace/4_eval_offline.ipynb: snippets/eval_offline.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/eval_offline.md \
				-o workspace/4_eval_offline.ipynb  
	sed -i 's/attachment://g' workspace/4_eval_offline.ipynb

5_delete.ipynb: snippets/delete.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/delete.md \
				-o 5_delete.ipynb  
	sed -i 's/attachment://g' 5_delete.ipynb