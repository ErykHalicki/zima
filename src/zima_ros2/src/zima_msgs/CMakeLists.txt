cmake_minimum_required(VERSION 3.8)
project(zima_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ServoCommand.msg"
  "msg/MotorCommand.msg"
  "msg/GripperCommand.msg"
  "msg/EncoderData.msg"
  "msg/ServoPositions.msg"
  "msg/HardwareCommand.msg"
  "msg/HardwareStatus.msg"
  "msg/ArmStateDelta.msg"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

ament_package()
