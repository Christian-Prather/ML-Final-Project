import grpc

import ptolemy_pb2 as sensors
import ptolemy_pb2_grpc as sensors_pb2_grpc

def hello():
    helloCommand = sensors.StreamRequest()
    response = stub.TestStream(helloCommand)
    print(response)

def imu():
    imuCommand = sensors.StreamRequest()
    response = stub.ImuStream(imuCommand)
    for message in response:
        if message.quaternion_x:
            print("Quaternion X:", message.quaternion_x)
        if message.quaternion_y:
            print("Quaternion Y:", message.quaternion_y)
        if message.quaternion_z:
            print("Quaternion Z:", message.quaternion_z)
        if message.quaternion_w:
            print("Quaternion W:", message.quaternion_w)
        if message.acceleration_x:
            print("Acceleration X:", message.acceleration_x)
        if message.acceleration_y:
            print("Acceleration Y:", message.acceleration_y)
        if message.acceleration_z:
            print("Acceleration Z:", message.acceleration_z)
        if message.gyro_x:
            print("Gyro X:", message.gyro_x)
        if message.gyro_y:
            print("Gyro Y:", message.gyro_y)
        if message.gyro_z:
            print("Gyro Z:", message.gyro_z)
        print("##################################################")
        if message.euler_x:
            print("Euler X", message.euler_x)
        if message.euler_y:
            print("Euler Y", message.euler_y)
        if message.euler_z:
            print("Euler Z", message.euler_z)

if __name__ == '__main__':

    channel = grpc.insecure_channel('localhost:1997')
    stub = sensors_pb2_grpc.SensorsStub(channel)

    # hello()
    imu()
    
