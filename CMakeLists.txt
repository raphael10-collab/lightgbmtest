cmake_minimum_required(VERSION 3.15...3.31)
project(lightgbmtest)

add_executable(${PROJECT_NAME}
    ./src/main.cpp
)

target_link_directories(${PROJECT_NAME} PUBLIC
    ./src/LightGBM/lib
)

target_include_directories(${PROJECT_NAME} PUBLIC

    ./src/LightGBM/include

)

target_link_libraries (${PROJECT_NAME} PUBLIC
    _lightgbm
)
