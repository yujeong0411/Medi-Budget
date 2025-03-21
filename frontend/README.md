## 개발환경 설정

1. 프로젝트 환경설정(vite 기반 React 프로젝트) 설치 : `npm install vite@latest` <br />
2. React Router 설치 : `npm install react-router-dom` <br/>
3. React 중앙집중식 상태관리 라이브러리 Zustand 설치 : `npm install zustand` <br/>
4. 외부 open API 통신을 위한 라이브러리 axios 설치 : `npm install axios` <br/>
5. Carousel 설치 : `npm install react-slick --save`, `npm install slick-carousel --save`<br/>
6. tailwindcss 설치 : `npm install -D tailwindcss@3.3.5 postcss@8.4.31 autoprefixer@10.4.15` <br/>
7. Typescript에서 Node.js 모듈을 쓸 수 있도록 환경 구축 : `npm install @types/node` <br/>
8. React Toasts Popup 모듈 설치 : `npm install react-toastify ` <br/>
9. ESLint 설치 및 설정 :
   `npm install -D eslint eslint-plugin-react @typescript-eslint/parser @typescript-eslint/eslint-plugin` <br/>
10. Prettier 설치 및 설정 : `npm install -D prettier eslint-config-prettier eslint-plugin-prettier` <br/>
11. shadcn-ui 설치 및 설정 : `npx shadcn-ui@latest init`,
    `npm install tailwindcss-animate class-variance-authority clsx tailwind-merge lucide-react`

### Prettier 설정 방법

- WebStorm에서 Prettier 설정:

1. Settings → Languages & Frameworks → JavaScript → Prettier 이동
2. 'Automatic Prettier configuration' 선택
3. 설치된 Prettier 패키지와 설정 파일이 자동으로 인식되어 사용됨
