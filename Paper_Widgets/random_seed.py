{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "usage: ipykernel_launcher.py [-h] [--seed SEED] [--n_gpu N_GPU]\n",
      "ipykernel_launcher.py: error: unrecognized arguments: --f=/root/.local/share/jupyter/runtime/kernel-v2-112467Cgiw2cI2c5JJ.json\n"
     ]
    },
    {
     "ename": "SystemExit",
     "evalue": "2",
     "output_type": "error",
     "traceback": [
      "An exception has occurred, use %tb to see the full traceback.\n",
      "\u001b[0;31mSystemExit\u001b[0m\u001b[0;31m:\u001b[0m 2\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.10/dist-packages/IPython/core/interactiveshell.py:3561: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.\n",
      "  warn(\"To exit: use 'exit', 'quit', or Ctrl-D.\", stacklevel=1)\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import torch\n",
    "import random\n",
    "import argparse\n",
    "\n",
    "def set_seed(args):\n",
    "    np.random.seed(args.seed)\n",
    "    torch.manual_seed(args.seed)\n",
    "    if args.n_gpu > 0:\n",
    "        torch.cuda.manual_seed_all(args.seed)\n",
    "    random.seed(args.seed)  # 额外设置 Python 内置的随机数生成器的种子\n",
    "\n",
    "# 定义参数解析\n",
    "parser = argparse.ArgumentParser()\n",
    "parser.add_argument(\"--seed\", type=int, default=42, help=\"random seed for initialization\")\n",
    "parser.add_argument(\"--n_gpu\", type=int, default=1, help=\"number of GPUs to use\")\n",
    "args = parser.parse_args()\n",
    "\n",
    "# 设置随机种子\n",
    "set_seed(args)\n",
    "\n",
    "# 生成一些随机数\n",
    "print(\"NumPy 随机数:\", np.random.rand())\n",
    "print(\"PyTorch 随机数 (CPU):\", torch.rand(1))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
