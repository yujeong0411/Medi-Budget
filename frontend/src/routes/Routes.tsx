import HomePage from "@/pages/HomePage";
import { Route, Routes, BrowserRouter } from "react-router-dom";

const AppRoutes = () => {
  return(
    <BrowserRouter>
    <Routes>
      <Route path="/" element={<HomePage />}/>
    </Routes>
    </BrowserRouter>
  )
}

export default AppRoutes