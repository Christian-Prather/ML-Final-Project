Rails.application.routes.draw do
  resources :jobs do
    member do
      get :run
    end
  end
  root 'jobs#index'
  # For details on the DSL available within this file, see https://guides.rubyonrails.org/routing.html
end
