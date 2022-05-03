echo "请确认是否已shutdown所有ipykernel"
read -t 5 -p "按任意键继续..."
sudo systemctl stop jetcard_jupyter
sudo systemctl restart nvargus-daemon
sudo systemctl start jetcard_jupyter
