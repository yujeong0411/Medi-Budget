import Logo from '@/components/logo.tsx';
import logo from '@/assets/logo.png'
import { useRef, useEffect } from 'react';

function HomePage() {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    // 비디오 요소가 있을 때
    if (videoRef.current) {
      // 비디오 로드 후 재생 시작
      videoRef.current.load();
    }
  }, []);
  return (
    <div className="w-full h-full relative">
      {/*헤더*/}
      <header className="fixed left-0 top-0 w-full bg-white z-10 shadow-md">
        <div className="pl-20 w-full h-20 flex items-center px-4">
          <Logo />
          <div className="pl-4 text-4xl font-medium">Medi Budget</div>
        </div>
      </header>

      {/*비디오*/}
      <section className="relative bg-black/60 text-center min-h-[800px] p-12">
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
            <h2 className="text-3xl font-bold mb-8">병원 방문 전, 이런 고민 해보셨나요?</h2>
            <div className="text-2xl font-bold mb-8">
              <p>"이 증상으로 병원에 가면 진료비가 얼마나 나올까?"</p>
              <p>"독감진료는 얼마를 내지?"</p>
              <p>"아이는 감기 진료가 얼마나 나올지 모르겠어..."</p>
            </div>
            <p className="text-2xl mt-8">
              Medi Budget은 진료비의 불확실성을 해소하고, 의료비에 대한 투명성을 높여 합리적인 의료 소비를 돕습니다.
            </p>
          </div>
        </div>
      </section>

      {/*검색 창*/}
     <section className="w-full bg-indigo-">

     </section>
    </div>
  );
}

export default HomePage;
