#pragma once

float clamp(float value, float min, float max){
    if(value < min)
        return min;
    if(value > max)
        return max;
    return value;
}

float clamp01(float value){
    return clamp(value, 0, 1);
}