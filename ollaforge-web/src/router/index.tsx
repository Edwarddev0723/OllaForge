import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import { MainLayout } from '../layouts';
import { GeneratePage, AugmentPage, ConfigPage } from '../pages';

const router = createBrowserRouter([
  {
    path: '/',
    element: <MainLayout />,
    children: [
      {
        index: true,
        element: <GeneratePage />,
      },
      {
        path: 'augment',
        element: <AugmentPage />,
      },
      {
        path: 'config',
        element: <ConfigPage />,
      },
    ],
  },
]);

export const AppRouter: React.FC = () => {
  return <RouterProvider router={router} />;
};

export default AppRouter;
