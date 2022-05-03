import logging
import serial.tools.list_ports
import serial
from time import sleep


def _map(x, in_min, in_max, out_min, out_max):
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)


# 发送Y并获取通道标准值
def getChannelValue(mySerial, arduino_standardization=True, standardization=True, max_value=2000, min_value=1000, CH1_bias=0, CH2_bias=0, CH3_bias=0):
    if arduino_standardization:
        mySerial.sendData(data="Y")
        return mySerial.getData(split_str="#")
    else:
        mySerial.sendData(data="L")
        if standardization:
            data = mySerial.getData(split_str="#")
            output_data = []
            for a in range(len(data)):
                value = int(data[a])
                if a == 0:
                    value -= CH1_bias
                elif a == 1:
                    value -= CH2_bias
                elif a == 2:
                    value -= CH3_bias
                else:
                    return ValueError
                if value > 2000:
                    value = 2000
                elif value < 1000:
                    value = 1000
                elif abs(value - 1500) < 50:
                    value = 1500
                output_data.append(_map(value, min_value, max_value, -100, 100) / 100)
            return output_data
        else:
            return mySerial.getData(split_str="#")


class SerialObject:
    """
    Allow to transmit data to a Serial Device like Arduino.
    Example send $255255000
    """

    def __init__(self, portNo=None, baudRate=9600):
        """
        Initialize the serial object.
        :param portNo: Port Number.
        :param baudRate: Baud Rate.
        """
        self.portNo = portNo
        self.baudRate = baudRate
        connected = False
        if self.portNo is None:
            ports = list(serial.tools.list_ports.comports())
            for p in ports:
                if "Arduino" in p.description:
                    print('{} Connected'.format(p.description))
                    self.ser = serial.Serial(p.device)
                    self.ser.baudrate = baudRate
                    connected = True
            if not connected:
                logging.warning("Arduino Not Found. Please enter COM Port Number instead.")

        else:
            try:
                self.ser = serial.Serial(self.portNo, self.baudRate)
                # print("Serial Device Connected")
                print("参数设置: 串口=%s, 波特率=%d" % (self.portNo, self.baudRate))
            except:
                logging.warning("Serial Device Not Connected")
                exit()

    def sendData(self, data):
        """
        Send data to the Serial device
        :param data: str to send
        """

        try:
            self.ser.write(data.encode("utf-8"))
            self.ser.write("\r\n".encode("utf-8"))
            # print("Successful write {} to {}".format(data, self.portNo))
            return True
        except:
            return False

    def getData(self, split_str="#"):
        """
        :param split_str: the special str used to split the return str
        :return: list of data received
        """
        data = self.ser.readline()
        data = data.decode("utf-8")
        data = data.split(split_str)
        data[-1] = data[-1].strip()
        return data


def link_test_main():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        # print(p.description)
        if "Arduino" in p.description:
            print("Arduino at {}".format(p.description))


def value_take_main(loop_num=500):
    mySerial = SerialObject(portNo="/dev/ttyACM0", baudRate=9600)
    sleep(2)
    for a in range(loop_num):
        value = getChannelValue(mySerial=mySerial, arduino_standardization=False, standardization=True)
        sleep(0.1)
        print(value)


if __name__ == '__main__':
    # link_test_main()
    value_take_main()
