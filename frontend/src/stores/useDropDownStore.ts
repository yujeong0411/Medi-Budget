import { create } from 'zustand/react';
import { fetchDiseases, fetchRegions, fetchHospitalTypes } from '@/services/dropDown.services.ts';
import {DropDown} from '@/types/dropdown.types.ts';

const useDropDownStore = create<DropDown>((set) => ({
  diseases: [],
  hospitalTypes: [],
  regions: [],

  selectedDisease: null,
  selectedHospitalType: null,
  selectedRegion: null,
  age: '',

  loading: {
    diseases: false,
    hospitalTypes: false,
    regions: false,
  },

  error: {
    diseases: null,
    hospitalTypes: null,
    regions: null,
  },

  setSelectedDisease: (disease_id) => set({ selectedDisease: disease_id }),
  setSelectedHospitalType: (hospital_type_id) => set({ selectedHospitalType: hospital_type_id }),
  setSelectedRegion: (region_id) => set({ selectedRegion: region_id }),
  setAge: (age) => set({ age }),

  fetchAllData: async () => {
    // 질병 데이터 가져오기
    set((state) => ({
      loading: { ...state.loading, diseases: true },
      error: { ...state.error, diseases: null }
    }));
  try {
    const diseases = await fetchDiseases()
    set((state) => ({
      diseases,
      loading: {...state.loading, diseases:false },
    }))
  } catch (error: any) {
    set((state) => ({
      loading: {...state.loading, diseases:false},
      error: { ...state.error, diseases: error.message }
    }))
  }


    // 의료기관 종류 데이터 가져오기
    set((state) => ({
      loading: { ...state.loading, hospitalTypes: true },
      error: { ...state.error, hospitalTypes: null }
    }));

    try {
      const hospitalTypes = await fetchHospitalTypes();
      set((state) => ({
        hospitalTypes,
        loading: { ...state.loading, hospitalTypes: false }
      }));
    } catch (error: any) {
      set((state) => ({
        loading: { ...state.loading, hospitalTypes: false },
        error: { ...state.error, hospitalTypes: error.message }
      }));
    }

    // 지역 데이터 가져오기
    set((state) => ({
      loading: { ...state.loading, regions: true },
      error: { ...state.error, regions: null }
    }));

    try {
      const regions = await fetchRegions();
      set((state) => ({
        regions,
        loading: { ...state.loading, regions: false }
      }));
    } catch (error: any) {
      set((state) => ({
        loading: { ...state.loading, regions: false },
        error: { ...state.error, regions: error.message }
      }));
    }
  },


}))

export default useDropDownStore;
