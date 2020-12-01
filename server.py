import random
import grpc

import ptolemy_pb2_grpc
import ptolemy_pb2
import time
from concurrent import futures
from mpu6050 import mpu6050
sensor = mpu6050(0x68)
from detect import shared
import detect

class Sensors(ptolemy_pb2_grpc.SensorsServicer):
    def ImuStream(self, request_iterator, context):
        # def stream():
        #     while True:
        #         time.sleep(1)
        #         yield 
        # output_stream = stream()
        while True:
            data = ptolemy_pb2.ImuData()
            data.quaternion_x = (12)
            data.quaternion_y = (15)
            data.quaternion_z = (0)
            data.quaternion_w = (13)
            data.acceleration_x = (0)
            data.acceleration_y = (0)
            data.acceleration_z = (0)

            data.gyro_x=(420)
            data.gyro_y=(420)
            data.gyro_z=(420)

            data.euler_x=(555)
            data.euler_y=(555)
            data.euler_z=(555)

            data.acceleration_x = sensor.get_accel_data()['x']
            data.acceleration_y=(200)
            data.acceleration_z=(200)

            data.magnetometer_x=(6)
            data.magnetometer_y=(6)
            data.magnetometer_z=(6)

            data.gravity_x=(27)
            data.gravity_y=(27)
            data.gravity_z=(27)
            data.isKeyPoint = shared.keyPoint
            print("Imu requested...")
            time.sleep(2)
            yield data




if __name__ == "__main__":
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ptolemy_pb2_grpc.add_SensorsServicer_to_server(Sensors(), server)

    server.add_insecure_port('10.0.97.28:1997')
    server.start()
    print("Listening...")
    detect.main()
    server.wait_for_termination()
