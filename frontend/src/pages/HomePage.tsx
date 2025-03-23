import Logo from '@/components/logo.tsx';
import { useRef, useEffect, useState } from 'react';
import useDropDownStore from '@/stores/useDropDownStore.ts';
import DropDown from '@/components/DropDown.tsx';

function HomePage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [showScrollTop, setShowScrollTop] = useState(false);

  // Zustand 스토어에서 상태와 액션 가져오기
  const {
    diseases,
    hospitalTypes,
    regions,

    selectedDisease,
    selectedHospitalType,
    selectedRegion,
    age,

    loading,
    error,

    setSelectedDisease,
    setSelectedHospitalType,
    setSelectedRegion,
    setAge,
    fetchAllData,
  } = useDropDownStore();

  useEffect(() => {
    // 비디오 요소가 있을 때
    if (videoRef.current) {
      // 비디오 로드 후 재생 시작
      videoRef.current.load();
    }

    // 스크롤 이벤트
    const handleScroll = () => {
      // 스크롤 위치가 300px 이상이면 표시
      if (window.scrollY > 300) {
        setShowScrollTop(true);
      } else {
        setShowScrollTop(false);
      }
    };

    // 스크롤 이벤트
    window.addEventListener('scroll', handleScroll);

    // api 가져오기
    fetchAllData();

    // 컴포넌트 unmonut 시 이벤트 리스너 제거
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, [fetchAllData]);

  // 상단으로 스크롤하는 함수
  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // 검색 핸들러
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // api 호출
  };


  const diseaseOptions = diseases.map(disease => ({
    id: disease.disease_id,
    name: disease.disease_name
  }));

  const hospitalTypeOptions = hospitalTypes.map(hospitalType => ({
    id: hospitalType.hospital_type_id,
    name: hospitalType.hospital_type
  }));

  const regionOptions = regions.map(region => ({
    id: region.region_id,
    name: region.region_name
  }));


  return (
    <div className="w-full h-full relative">
      {/*헤더*/}
      <header className="fixed left-0 top-0 w-full bg-white z-10 shadow-md">
        <div className="pl-[4rem] w-full h-[3.5rem] flex items-center px-4">
          <Logo />
          <div className="text-baseColor pl-4 text-xl sm:text-3xl font-medium">Medi Budget</div>
        </div>
      </header>

      {/*비디오*/}
      <section className="relative bg-black/60 text-center min-h-[600px] sm:min-h-[700px] md:min-h-[800px] p-4 sm:p-8 md:p-12 flex items-center justify-center">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          className="absolute top-0 left-0 w-full h-full object-cover z-[-1]"
        >
          {/*<source src={videoFile} type="video/mp4" />*/}
          "Your browser does not support the video tag."
        </video>

        {/* Dark Overlay */}
        <div className="absolute top-0 left-0 w-full h-full bg-black opacity-60 z-1"></div>
        {/*비디오 위에 표시할 콘텐츠*/}
        <div className="relative z-2 flex flex-col items-center justify-center h-full text-white p-4">
          <div className="max-w-4xl mx-auto text-center">
            <h2 className="text-xl sm:text-2xl md:text-3xl font-bold mb-4 sm:mb-6 md:mb-8">
              병원 방문 전, 이런 고민 해보셨나요?
            </h2>
            <div className="text-lg sm:text-xl md:text-2xl font-bold mb-4 sm:mb-6 md:mb-8">
              <p className="mb-2">"이 증상으로 병원에 가면 진료비가 얼마나 나올까?"</p>
              <p className="mb-2">"독감진료는 얼마를 내지?"</p>
              <p>"아이는 감기 진료가 얼마나 나올지 모르겠어..."</p>
            </div>
            <p className="text-base sm:text-xl md:text-2xl mt-4 sm:mt-6 md:mt-8">
              Medi Budget은 진료비의 불확실성을 해소하고, 의료비에 대한 투명성을 높여 합리적인 의료
              소비를 돕습니다.
            </p>
          </div>
        </div>
      </section>

      {/*검색 창*/}
      <section className="w-full h-full bg-indigo-50 py-[5rem]">
        <div className="flex flex-col items-center justify-center">
          <div className="w-[6rem] sm:w-[8rem] md:w-[10rem] h-1 sm:h-1.5 md:h-2 bg-baseColor items-center justify-center rounded-lg mb-[3.5rem]"></div>
          <h2 className="text-xl sm:text-2xl md:text-3xl font-semibold mb-[2rem] sm:mb-[2.5rem] md:mb-[3rem] text-center px-4">
            Medi Budget으로 진료비를 예측해보세요.
          </h2>

          <form className="w-full max-w-4xl px-4">
            {/*첫번째 행*/}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-6 p-3">
              <div className="flex flex-col">
                <DropDown
                  id="disease"
                  options={diseaseOptions}
                  placeholder="질환을 검색하세요"
                  label="질환"
                  value={selectedDisease}
                  onChange={setSelectedDisease}
                  loading={loading.diseases}
                  error={error.diseases}
                  keyProperty="id"
                  displayProperty="name"
                />
              </div>
              <div className="flex flex-col">
                <label className="text-base sm:text-lg mb-1 sm:mb-2">나이</label>
                <input
                  id="age"
                  type="number"
                  min="0"
                  max="120"
                  placeholder="나이를 입력하세요."
                  className="p-3 rounded-lg min-h-[50px]"
                  value={age}
                  onChange={(e) => setAge(e.target.value)}
                />
              </div>
            </div>

            {/*두번째 행*/}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-6 p-3">
              <div className="flex flex-col">
                <DropDown
                  id="hospitalType"
                  options={hospitalTypeOptions}
                  placeholder="병원 종류를 검색하세요"
                  label="병원 종류"
                  value={selectedHospitalType}
                  onChange={setSelectedHospitalType}
                  loading={loading.hospitalTypes}
                  error={error.hospitalTypes}
                  keyProperty="id"
                  displayProperty="name"
                />
              </div>

                <div className="flex flex-col">
                  <DropDown
                    id="region"
                    options={regionOptions}
                    placeholder="지역을 검색하세요"
                    label="지역"
                    value={selectedRegion}
                    onChange={setSelectedRegion}
                    loading={loading.regions}
                    error={error.regions}
                    keyProperty="id"
                    displayProperty="name"
                  />
                </div>
            </div>

            {/*버튼*/}
            <div className="flex sm:justify-end mt-6">
              <button
                type="submit"
                className="  w-full sm:w-auto bg-baseColor text-base sm:text-lg text-white font-semibold rounded-md px-4 py-[0.8rem]"
              >
                진료비 예측하기
              </button>
            </div>
          </form>
        </div>

        {/*결과*/}
        <div className="w-full mt-12 sm:mt-16 md:mt-20 px-4">
          <div className="flex flex-col items-center justify-center space-y-[2rem] sm:space-y-[3rem] md:space-y-[4rem]">
            {/*첫번째 행*/}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 sm:gap-10 md:gap-20 w-full max-w-6xl">
              {/*1일 예상 진료비*/}
              <div className="flex flex-col">
                <h3 className="text-lg sm:text-xl mb-2 sm:mb-3">1일 예상 진료비</h3>
                <div className="bg-white rounded-lg p-5">
                  <div className="text-xl sm:text-2xl md:text-3xl font-bold">3000~12000(천원)</div>
                </div>
              </div>

              {/*평균 내원 일수 및 총 예상 진료비*/}
              <div className="flex flex-col">
                <h3 className="text-lg sm:text-xl mb-2 sm:mb-3">
                  평균 외래 내원일수 및 예상 진료비
                </h3>
                <div className="bg-white rounded-lg p-5">
                  <div className="text-xl sm:text-2xl md:text-3xl font-bold">3000~12000(천원)</div>
                </div>
              </div>
            </div>

            {/*두번째 행*/}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 sm:gap-10 md:gap-20 w-full max-w-6xl">
              {/*각 연령대별 해당 질환의 평균 진료 내원일수*/}
              <div className="flex flex-col">
                <h3 className="text-lg sm:text-xl mb-2 sm:mb-3">
                  연령대별 해당 질환의 평균 진료 내원일수
                </h3>
                <div className="bg-white rounded-lg p-5">
                  <div className="text-xl sm:text-2xl md:text-3xl font-bold">3000~12000(천원)</div>
                </div>
              </div>

              {/*해당 연령대별 외래 진료 질환 순위*/}
              <div className="flex flex-col">
                <h3 className="text-lg sm:text-xl mb-2 sm:mb-3">연령대별 외래 진료 질환 순위</h3>
                <div className="bg-white rounded-lg p-5">
                  <div className="text-xl sm:text-2xl md:text-3xl font-bold">3000~12000(천원)</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/*Footer*/}
      <footer className="w-full bg-gray-700 py-4 sm:py-6">
        <div className="container mx-auto px-4">
          <div className="flex flex-col sm:flex-row justify-between items-center text-gray-400">
            <div className="text-xs sm:text-sm mb-4 sm:mb-0">
              Copyright medibudget © Corp. All rights reserved.
            </div>
            <div className="flex items-center space-x-2 sm:space-x-4 text-xs">
              <span>개인정보처리방침</span>
              <span className="hidden sm:inline-block border-r border-zinc-200 h-3"></span>
              <span>서비스이용약관</span>
              <span className="hidden sm:inline-block border-r border-zinc-200 h-3"></span>
              <span>위치기반서비스이용약관</span>
            </div>
          </div>
        </div>
      </footer>

      {/*스크롤 버튼*/}
      {showScrollTop && (
        <button
          onClick={scrollToTop}
          className="fixed bottom-6 right-6 bg-baseColor hover:bg-indigo-900 text-white w-12 h-12 rounded-full flex items-center justify-center shadow-lg transition-all duration-300 z-50"
          aria-label="위로 이동"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
          </svg>
        </button>
      )}
    </div>
  );
}

export default HomePage;
