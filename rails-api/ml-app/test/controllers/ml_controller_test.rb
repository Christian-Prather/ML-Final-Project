require 'test_helper'

class MlControllerTest < ActionDispatch::IntegrationTest
  test "should get index" do
    get ml_index_url
    assert_response :success
  end

end
