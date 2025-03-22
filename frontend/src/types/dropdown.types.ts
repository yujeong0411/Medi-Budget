export interface Disease {
  disease_id: number;
  disease_name: string;
  disease_code: string;
}

export interface HospitalType {
  hospital_type_id: number;
  hospital_type: string;
}

export interface Region {
  region_id: number;
  region_name: string;
}

export interface DropDown {
  diseases: Disease[];
  hospitalTypes: HospitalType[];
  regions: Region[];

  // 선택된 값
  selectedDisease: number | null;
  selectedHospitalType: number | null;
  selectedRegion: number | null;
  age: string;

  // 로딩 상태
  loading: {
    diseases: boolean;
    hospitalTypes: boolean;
    regions: boolean;
  };

  // 에러 상태
  error: {
    diseases: string | null;
    hospitalTypes: string | null;
    regions: string | null;
  };

  // 액션
  setSelectedDisease: (id: number | null) => void;
  setSelectedHospitalType: (id: number | null) => void;
  setSelectedRegion: (id: number | null) => void;
  setAge: (age: string) => void;
  fetchAllData: () => Promise<void>;
}