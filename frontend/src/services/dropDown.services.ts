import {Disease, Region, HospitalType} from '@/types/dropdown.types.ts';
import { axiosInstance } from '@/services/axios.ts';

export const fetchDiseases = async (): Promise<Disease[]> => {
  try {
    const response = await axiosInstance.get('/diseases')
    return response.data;
  } catch (error) {
    console.error("질병 api 호출 에러:", error);
    throw error;
  }
}

export const fetchHospitalTypes = async (): Promise<HospitalType[]> => {
  try {
    const response = await axiosInstance.get('/hospital-types')
    return response.data;
  } catch (error) {
    console.error("병원종류 api 호출 에러:", error);
    throw error;
  }
}

export const fetchRegions = async (): Promise<Region[]> => {
  try {
    const response = await axiosInstance.get('/regions')
    return response.data;
  } catch (error) {
    console.error("지역 api 호출 에러:", error);
    throw error;
  }
}