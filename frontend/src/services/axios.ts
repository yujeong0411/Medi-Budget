import axios from 'axios';

export const axiosInstance = axios.create({
  baseURL: '/api',
  timeout: 5000,
  withCredentials: true, // 쿠키를 포함시키기 위해 설정한다

})
