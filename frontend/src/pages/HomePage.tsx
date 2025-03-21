import Logo from '@/components/logo.tsx';
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
          <source src={videoFile} type="video/mp4" />
          "Your browser does not support the video tag."
        </video>
        {/*비디오 위에 표시할 콘텐츠*/}
        <div>
          <h1>텍스트</h1>
        </div>
      </section>
    </div>
  );
}

export default HomePage;
